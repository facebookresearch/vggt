"""
Concurrent Depth Anything TensorRT inference helpers.

This module provides a small worker pool that loads multiple copies of the
TensorRT engine and runs them in parallel threads.  Each worker owns its CUDA
context which keeps the call-site free from threading concerns.
"""
from __future__ import annotations

import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from . import io_utils, trt_utils

try:  # pragma: no cover - optional deps validated at runtime
    import tensorrt as trt  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("TensorRT is required for Depth Anything inference.") from exc

try:
    import pycuda.driver as cuda  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyCUDA is required for Depth Anything inference.") from exc


def _vol(shape: Iterable[int]) -> int:
    prod = 1
    for dim in shape:
        prod *= int(dim)
    return prod


def _trt_dtype_to_np(dt: "trt.DataType") -> np.dtype:
    if dt == trt.DataType.FLOAT:
        return np.float32
    if dt == trt.DataType.HALF:
        return np.float16
    if dt == trt.DataType.BF16:
        return np.dtype("bfloat16")
    if dt == trt.DataType.INT8:
        return np.int8
    if dt == trt.DataType.INT32:
        return np.int32
    raise ValueError(f"Unsupported TensorRT dtype: {dt}")


@dataclass
class _Task:
    tensor: np.ndarray
    orig_hw: Tuple[int, int]
    future: Future


class _DepthWorker(threading.Thread):
    """
    Background worker that owns a TensorRT execution context.
    """

    def __init__(
        self,
        engine_path: Path,
        device_index: int,
        ctrl_queue: "queue.Queue[Tuple[str, object]]",
    ) -> None:
        super().__init__(daemon=True)
        self.engine_path = Path(engine_path)
        self.device_index = int(device_index)
        self.ctrl_queue = ctrl_queue
        self.tasks: "queue.Queue[Optional[_Task]]" = queue.Queue(maxsize=8)
        self._stop_event = threading.Event()

    def submit(self, task: _Task) -> None:
        self.tasks.put(task)

    def stop(self) -> None:
        self._stop_event.set()
        self.tasks.put(None)

    # pylint: disable=too-many-locals
    def run(self) -> None:  # pragma: no cover - exercised in integration
        cuda.init()
        device = cuda.Device(self.device_index)
        context = device.make_context()
        stream = None
        # Keep references initialised for deterministic cleanup order
        runtime = None
        engine = None
        context_trt = None
        host_input = None
        device_input = None
        host_outputs = []
        device_outputs = []
        try:
            logger = trt.Logger(trt.Logger.ERROR)
            with open(self.engine_path, "rb") as handle:
                runtime = trt.Runtime(logger)
                engine = runtime.deserialize_cuda_engine(handle.read())
            if engine is None:
                raise RuntimeError(f"Failed to deserialize engine {self.engine_path}")

            use_io_tensors = hasattr(engine, "num_io_tensors")
            if use_io_tensors:
                input_name = None
                output_names = []
                for i in range(int(engine.num_io_tensors)):
                    name = engine.get_tensor_name(i)
                    mode = engine.get_tensor_mode(name)
                    if mode == trt.TensorIOMode.INPUT:
                        input_name = name
                    else:
                        output_names.append(name)
                if input_name is None or not output_names:
                    raise RuntimeError("Unexpected IO tensor layout in Depth Anything engine.")
                input_shape = tuple(engine.get_tensor_shape(input_name))
                input_dtype = _trt_dtype_to_np(engine.get_tensor_dtype(input_name))
                output_meta = [
                    (name, tuple(engine.get_tensor_shape(name)), _trt_dtype_to_np(engine.get_tensor_dtype(name)))
                    for name in output_names
                ]
            else:
                input_index = None
                output_indices = []
                for i in range(int(engine.num_bindings)):
                    if engine.binding_is_input(i):
                        input_index = i
                    else:
                        output_indices.append(i)
                if input_index is None or not output_indices:
                    raise RuntimeError("Unexpected binding layout in Depth Anything engine.")
                input_shape = tuple(engine.get_binding_shape(input_index))
                input_dtype = _trt_dtype_to_np(engine.get_binding_dtype(input_index))
                output_meta = [
                    (i, tuple(engine.get_binding_shape(i)), _trt_dtype_to_np(engine.get_binding_dtype(i)))
                    for i in output_indices
                ]

            context_trt = engine.create_execution_context()
            if context_trt is None:
                raise RuntimeError("Failed to create TensorRT execution context.")

            if use_io_tensors and hasattr(context_trt, "set_input_shape"):
                context_trt.set_input_shape(input_name, input_shape)

            stream = cuda.Stream()
            host_input = cuda.pagelocked_empty(_vol(input_shape), input_dtype).reshape(input_shape)
            device_input = cuda.mem_alloc(host_input.nbytes)
            host_outputs = [
                cuda.pagelocked_empty(_vol(shape), dtype).reshape(shape)
                for _, shape, dtype in output_meta
            ]
            device_outputs = [cuda.mem_alloc(buf.nbytes) for buf in host_outputs]

            # Share metadata back to controller
            self.ctrl_queue.put(
                (
                    "meta",
                    {
                        "input_shape": input_shape,
                        "input_dtype": input_dtype,
                        "output_shape": host_outputs[0].shape if host_outputs else None,
                    },
                )
            )

            while not self._stop_event.is_set():
                task = self.tasks.get()
                if task is None:
                    break
                tensor, orig_hw, future = task.tensor, task.orig_hw, task.future
                try:
                    np.copyto(host_input, tensor, casting="no")
                    if use_io_tensors:
                        context_trt.set_tensor_address(input_name, int(device_input))
                        for (name, _, _), dev in zip(output_meta, device_outputs):
                            context_trt.set_tensor_address(name, int(dev))
                    cuda.memcpy_htod_async(device_input, host_input, stream)
                    if use_io_tensors:
                        ok = context_trt.execute_async_v3(stream.handle)
                    else:
                        bindings = [None] * int(engine.num_bindings)
                        bindings[input_index] = int(device_input)
                        for (idx, _, _), dev in zip(output_meta, device_outputs):
                            bindings[idx] = int(dev)
                        ok = context_trt.execute_async_v2(bindings, stream.handle)
                    if not ok:
                        raise RuntimeError("TensorRT execution failed.")
                    for host_buf, dev_buf in zip(host_outputs, device_outputs):
                        cuda.memcpy_dtoh_async(host_buf, dev_buf, stream)
                    stream.synchronize()
                    output = host_outputs[0].astype(np.float32, copy=True)
                    future.set_result((output, orig_hw))
                except Exception as exc:  # pragma: no cover - propagate to caller
                    stream.synchronize()
                    future.set_exception(exc)
        finally:
            # Ensure proper destruction order: sync stream, free TRT/CUDA buffers, then pop context
            try:
                if stream is not None:
                    try:
                        stream.synchronize()
                    except Exception:
                        pass
                # Explicitly delete device and host buffers before popping context
                try:
                    # Device allocations
                    for dev in device_outputs or []:
                        try:
                            del dev
                        except Exception:
                            pass
                    device_outputs = []
                    if device_input is not None:
                        try:
                            del device_input
                        except Exception:
                            pass
                        device_input = None
                    # Host pinned buffers (must be freed before context pop)
                    for buf in host_outputs or []:
                        try:
                            del buf
                        except Exception:
                            pass
                    host_outputs = []
                    if host_input is not None:
                        try:
                            del host_input
                        except Exception:
                            pass
                        host_input = None
                finally:
                    # Destroy TRT objects prior to context pop
                    try:
                        del context_trt
                    except Exception:
                        pass
                    try:
                        del engine
                    except Exception:
                        pass
                    try:
                        del runtime
                    except Exception:
                        pass
            finally:
                try:
                    context.pop()
                except Exception:
                    pass


class DepthAnythingPool:
    """
    Threaded pool around Depth Anything TensorRT engines.
    """

    @staticmethod
    def suggest_workers(backend: str) -> int:
        backend = (backend or "").lower()
        if backend in {"tensorrt", "trt"}:
            try:
                import multiprocessing

                return max(1, min(4, multiprocessing.cpu_count() // 4 or 1))
            except Exception:  # pragma: no cover - fallback when psutil unavailable
                return 2
        # ONNX Runtime and other backends default to single-threaded pool
        return 1

    def __init__(
        self,
        engine_dir: Path,
        *,
        precision: str = "auto",
        num_workers: int,
        backend: str = "tensorrt",
        device_index: int = 0,
        explicit_engine: Optional[Path] = None,
        providers: Optional[Tuple[str, ...]] = None,
    ) -> None:
        self.backend = backend
        self.providers = providers
        self.precision = precision
        if explicit_engine is not None:
            engine_path = Path(explicit_engine)
            if not engine_path.exists():
                raise FileNotFoundError(f"Depth Anything engine not found: {engine_path}")
            self.engine_path = engine_path
        else:
            engines = trt_utils.discover_engines(Path(engine_dir), precision_hint=None, base_name="depth")
            if not engines:
                raise RuntimeError(f"No Depth Anything engines found in {engine_dir}")
            self.engine_path = trt_utils.select_engine(engines, precision)
        self.num_workers = max(1, num_workers)
        self.device_index = device_index
        self.ctrl_queue: "queue.Queue[Tuple[str, object]]" = queue.Queue()
        self.workers = [
            _DepthWorker(self.engine_path, device_index, self.ctrl_queue)
            for _ in range(self.num_workers)
        ]
        for worker in self.workers:
            worker.start()
        # Wait for first metadata packet
        meta: Optional[Dict[str, object]] = None
        for _ in range(self.num_workers):
            key, payload = self.ctrl_queue.get()
            if key == "meta" and isinstance(payload, dict):
                meta = payload
                break
        if meta is None:
            raise RuntimeError("Failed to initialize Depth Anything workers.")
        input_shape = meta.get("input_shape")
        if not isinstance(input_shape, tuple):
            raise RuntimeError("Depth Anything engine did not report input shape.")
        self.input_shape = input_shape  # (1, 3, H, W)
        self.input_dtype = np.float32  # enforce fp32 in preprocessing
        raw_output_shape = meta.get("output_shape")
        if isinstance(raw_output_shape, tuple) and len(raw_output_shape) >= 2:
            self.output_shape_hw = tuple(raw_output_shape[-2:])
        else:
            self.output_shape_hw = (input_shape[-2], input_shape[-1])

    @property
    def viewport_size(self) -> Tuple[int, int]:
        _, _, h, w = self.input_shape
        return h, w

    def _prepare_tensor(self, image: np.ndarray) -> np.ndarray:
        prep = image.astype(np.float32, copy=False)
        h, w = prep.shape[:2]
        target_h, target_w = self.viewport_size
        if h != target_h or w != target_w:
            if io_utils.Image is None:  # pragma: no cover
                raise RuntimeError("Pillow is required for resizing images.")
            pil = io_utils.Image.fromarray((prep * 255.0).clip(0, 255).astype(np.uint8))
            pil = pil.resize((target_w, target_h))
            prep = np.asarray(pil, dtype=np.float32) / 255.0
        chw = np.transpose(prep, (2, 0, 1))[None, ...]
        return chw.astype(np.float32, copy=False)

    def infer_batch(self, images: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        futures: Dict[str, Future] = {}
        # Round-robin submit to workers
        for idx, (cam_id, image) in enumerate(images.items()):
            tensor = self._prepare_tensor(image)
            orig_hw = image.shape[:2]
            future: Future = Future()
            task = _Task(tensor=tensor, orig_hw=orig_hw, future=future)
            worker = self.workers[idx % len(self.workers)]
            worker.submit(task)
            futures[cam_id] = future

        results: Dict[str, np.ndarray] = {}
        for cam_id, future in futures.items():
            depth_raw, orig_hw = future.result()
            if depth_raw.shape[-2:] != orig_hw:
                if io_utils.Image is None:  # pragma: no cover
                    raise RuntimeError("Pillow is required for resizing outputs.")
                pil = io_utils.Image.fromarray(depth_raw.squeeze().astype(np.float32))
                pil = pil.resize((orig_hw[1], orig_hw[0]))
                depth = np.asarray(pil, dtype=np.float32)
            else:
                depth = depth_raw.squeeze().astype(np.float32)
            results[cam_id] = depth
        return results

    def close(self) -> None:
        for worker in self.workers:
            worker.stop()
        for worker in self.workers:
            worker.join(timeout=1.0)

    def __enter__(self) -> "DepthAnythingPool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
