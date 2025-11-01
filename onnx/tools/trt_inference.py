#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized single-stream TensorRT runner for static-batch VGGT engines.

- Single context + single CUDA stream (low latency for 1×batch)
- Async execution by default (no extra host sync)
- Pinned host memory + device buffers
- Optional output deep-copy for safe reuse
- Exact batch-size validation
- Configurable normalization: none | unit | imagenet
- BF16 support for RTX 5090
- CUDA event timings for H2D/compute/D2H breakdown
- Context-manager friendly (explicit cleanup)
- Multi-threaded CPU decode optimization

Examples:
  # Pure model benchmark (no I/O overhead)
  python trt_inference.py --engine model.engine --use-random --iters 200
  
  # Real images with async execution
  python trt_inference.py --engine model.engine --images-dir data/cams8 --norm imagenet --save-outputs out.npz
  
  # Detailed GPU timing breakdown
  python trt_inference.py --engine model.engine --use-random --cuda-events --verbose
"""
import os
import sys
import time
import argparse
import atexit
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

try:
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401 creates CUDA context
except Exception as e:
    print(f"[FATAL] PyCUDA not available: {e}", file=sys.stderr)
    sys.exit(1)

try:
    import tensorrt as trt
except Exception as e:
    print(f"[FATAL] TensorRT not available: {e}", file=sys.stderr)
    sys.exit(1)

try:
    import cv2
    # Optimize OpenCV threading for image decode
    cv2.setNumThreads(min(4, os.cpu_count() or 4))
except Exception:
    cv2 = None


# Cleanup handler to prevent shutdown spam
def _cleanup_cuda():
    """Cleanup CUDA context on exit to prevent error messages."""
    try:
        if hasattr(cuda, "Context"):
            cuda.Context.synchronize()
    except Exception:
        pass

atexit.register(_cleanup_cuda)


# ---------------- Utilities ----------------

def trt_to_np_dtype(dt: trt.DataType) -> np.dtype:
    """Map TensorRT dtype to NumPy dtype."""
    if dt == trt.DataType.FLOAT:  return np.float32
    if dt == trt.DataType.HALF:   return np.float16
    if dt == trt.DataType.BF16:   return np.dtype("bfloat16")  # Correct BF16 mapping
    if dt == trt.DataType.INT8:   return np.int8
    if dt == trt.DataType.INT32:  return np.int32
    if dt == trt.DataType.BOOL:   return np.bool_
    raise ValueError(f"Unsupported TRT dtype: {dt}")

def vol(shape: Tuple[int, ...]) -> int:
    """Compute volume of tensor shape."""
    v = 1
    for d in shape: v *= int(d)
    return int(v)

def pinned_empty(shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    """Allocate pinned (page-locked) host memory for fast H2D/D2H."""
    return cuda.pagelocked_empty(vol(shape), dtype).reshape(shape)

def ensure_c_contig(a: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Ensure array is C-contiguous and correct dtype."""
    if a.dtype != dtype: a = a.astype(dtype, copy=False)
    if not a.flags['C_CONTIGUOUS']: a = np.ascontiguousarray(a)
    return a


# ------------- Simple Runner ---------------

class SimpleTrtRunner:
    """
    Single-stream TensorRT inference runner optimized for RTX 5090.
    
    Features:
    - Async execution by default (no host sync overhead)
    - Pinned host buffers for fast transfers
    - Optional CUDA event timings
    - Optional output deep-copy for safe reuse
    """
    
    def __init__(
        self,
        engine_path: str,
        verbose: bool = False,
        force_sync: bool = False,
        device: int = 0
    ):
        """
        Initialize TensorRT runner.
        
        Args:
            engine_path: Path to .engine file
            verbose: Print detailed binding info
            force_sync: Force synchronous execution (not recommended)
            device: CUDA device ID
        """
        # Note: pycuda.autoinit already initialized device 0
        # For multi-GPU, use CUDA_VISIBLE_DEVICES before import
        self.device = device
        self.verbose = verbose
        
        self.logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self._load_engine(engine_path)
        
        if self.engine.has_implicit_batch_dimension:
            raise RuntimeError("Engine must be explicit-batch. Rebuild with EXPLICIT_BATCH.")
        
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")

        # --- Pick API family (bindings vs IO tensors) ---
        self._use_io_tensors = hasattr(self.engine, "num_io_tensors") and not hasattr(self.engine, "num_bindings")

        # Discover bindings (handle both TRT 8/9 and TRT 10+ APIs)
        self.bindings_meta: List[Dict[str, Any]] = []
        
        if self._use_io_tensors:
            # TRT ≥ 10: IO-tensor API
            mode_enum = trt.TensorIOMode  # INPUT / OUTPUT
            n = int(self.engine.num_io_tensors)
            for i in range(n):
                name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(name)
                is_input = (mode == mode_enum.INPUT)
                dtype = self.engine.get_tensor_dtype(name)
                np_dtype = trt_to_np_dtype(dtype)
                shape = tuple(self.engine.get_tensor_shape(name))
                fmt = self.engine.get_tensor_format(name) if hasattr(self.engine, "get_tensor_format") else trt.TensorFormat.LINEAR
                
                if any(d == -1 for d in shape):
                    raise RuntimeError(
                        f"Dynamic shape detected for tensor '{name}'. "
                        "This runner expects static shapes."
                    )
                
                if fmt != trt.TensorFormat.LINEAR and verbose:
                    print(f"[WARN] Tensor '{name}' uses non-LINEAR format: {fmt}")
                
                self.bindings_meta.append({
                    "index": i,  # index within IO-tensor list
                    "name": name,
                    "is_input": is_input,
                    "dtype": dtype,
                    "np_dtype": np_dtype,
                    "shape": shape,
                    "format": fmt,
                    "nbytes": np.dtype(np_dtype).itemsize * vol(shape),
                })
        else:
            # TRT 8/9: legacy binding API
            for i in range(int(self.engine.num_bindings)):
                name = self.engine.get_binding_name(i)
                is_input = self.engine.binding_is_input(i)
                dtype = self.engine.get_binding_dtype(i)
                np_dtype = trt_to_np_dtype(dtype)
                shape = tuple(self.engine.get_binding_shape(i))
                tensor_format = self.engine.get_binding_format(i)
                
                if any(d == -1 for d in shape):
                    raise RuntimeError(
                        f"Dynamic shape detected for binding '{name}'. "
                        "This runner expects static shapes."
                    )
                
                # Warn about non-linear formats
                if tensor_format != trt.TensorFormat.LINEAR and verbose:
                    print(f"[WARN] Binding '{name}' uses non-LINEAR format: {tensor_format}")
                
                self.bindings_meta.append({
                    "index": i,
                    "name": name,
                    "is_input": is_input,
                    "dtype": dtype,
                    "np_dtype": np_dtype,
                    "shape": shape,
                    "format": tensor_format,
                    "nbytes": np.dtype(np_dtype).itemsize * vol(shape)
                })

        # Separate input/output
        self.input_meta = [b for b in self.bindings_meta if b["is_input"]]
        self.output_meta = [b for b in self.bindings_meta if not b["is_input"]]
        
        if len(self.input_meta) != 1:
            print(f"[WARN] Expected exactly 1 input, found {len(self.input_meta)}.")
        self.input = self.input_meta[0]

        # Allocate single stream and buffers
        # --- Pick API family (bindings vs IO tensors) ---
        self._use_io_tensors = hasattr(self.engine, "num_io_tensors") and not hasattr(self.engine, "num_bindings")
        
        # Allocate single stream (prefer NON_BLOCKING if this PyCUDA exposes it)
        try:
            nb_flag = getattr(getattr(cuda, "stream_flags", None), "NON_BLOCKING", None)
            self.stream = cuda.Stream(flags=nb_flag) if nb_flag is not None else cuda.Stream()
        except TypeError:
            # Older PyCUDA builds may not accept the 'flags' kwarg
            self.stream = cuda.Stream()

        self._alloc_buffers()

        # Pick execution API based on TRT version
        if self._use_io_tensors:
            self._exec_api = "execute_async_v3"
        else:
            if force_sync:
                self._exec_api = "execute_v2" if hasattr(self.context, "execute_v2") else "execute_async_v2"
            else:
                # Default to async (no host sync overhead)
                self._exec_api = "execute_async_v2" if hasattr(self.context, "execute_async_v2") else "execute_v2"

        if verbose:
            print(f"[INFO] TRT version: {getattr(trt, '__version__', 'unknown')}")
            print(f"[INFO] Execution mode: {self._exec_api}")
            print("[INFO] Bindings:")
            for b in self.bindings_meta:
                io = "IN " if b["is_input"] else "OUT"
                print(f"  {io} {b['index']:2d} {b['name']:<32} {b['shape']} "
                      f"{b['np_dtype']} {b['nbytes']/1e6:.2f} MB")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        self.close()

    def close(self):
        """Explicitly cleanup CUDA resources."""
        try:
            if hasattr(self, 'stream'):
                self.stream.synchronize()
        except Exception:
            pass

    def _load_engine(self, path: str) -> trt.ICudaEngine:
        """Load serialized TensorRT engine."""
        with open(path, "rb") as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {path}")
        print(f"[INFO] Loaded engine: {path}")
        return engine

    def _alloc_buffers(self) -> None:
        """Allocate device buffers and pinned host buffers."""
        if self._use_io_tensors:
            # TRT 10+: Device buffers keyed by tensor name
            self.dev_ptrs_by_name: Dict[str, int] = {}
            for b in self.bindings_meta:
                dmem = cuda.mem_alloc(int(b["nbytes"]))
                self.dev_ptrs_by_name[b["name"]] = int(dmem)
                b["device_allocation"] = dmem  # keep reference alive

            # Tell the context where each tensor lives (set once, reuse forever)
            for b in self.bindings_meta:
                self.context.set_tensor_address(b["name"], self.dev_ptrs_by_name[b["name"]])

            # Pinned host buffers
            self.host_input = pinned_empty(self.input["shape"], self.input["np_dtype"])
            self.host_outputs: List[np.ndarray] = [
                pinned_empty(b["shape"], b["np_dtype"]) for b in self.output_meta
            ]
        else:
            # TRT 8/9: Legacy bindings path (indexed)
            self.dev_ptrs: List[int] = [0] * self.engine.num_bindings
            for b in self.bindings_meta:
                dmem = cuda.mem_alloc(int(b["nbytes"]))
                self.dev_ptrs[b["index"]] = int(dmem)
                b["device_allocation"] = dmem  # keep reference alive

            # Pinned host buffers
            self.host_input = pinned_empty(self.input["shape"], self.input["np_dtype"])
            self.host_outputs: List[np.ndarray] = [
                pinned_empty(b["shape"], b["np_dtype"]) for b in self.output_meta
            ]

    # ---------- Inference API ----------

    def infer(
        self,
        batch_nchw: np.ndarray,
        copy_outputs: bool = False
    ) -> List[np.ndarray]:
        """
        Run one inference.
        
        Args:
            batch_nchw: Input array with shape matching engine input
            copy_outputs: If True, return deep copies (safe for storage)
                         If False, return pinned buffers (faster, reused next call)
        
        Returns:
            List of output arrays (NumPy, host memory)
        """
        # Validate input shape/dtype/contiguity
        batch_nchw = ensure_c_contig(batch_nchw, self.input["np_dtype"])
        if tuple(batch_nchw.shape) != tuple(self.input["shape"]):
            raise ValueError(
                f"Input shape {tuple(batch_nchw.shape)} != "
                f"engine expected {self.input['shape']}"
            )

        if self._use_io_tensors:
            # --- TRT 10+: IO-tensor API path ---
            # H2D: Copy to pinned buffer then to device
            np.copyto(self.host_input, batch_nchw, casting='no')
            cuda.memcpy_htod_async(
                self.dev_ptrs_by_name[self.input["name"]],
                self.host_input,
                self.stream
            )

            # Execute (tensor addresses already set in _alloc_buffers)
            ok = self.context.execute_async_v3(self.stream.handle)
            if not ok:
                raise RuntimeError("TensorRT execute_async_v3 failed")

            # D2H: Copy outputs back to host
            outs: List[np.ndarray] = []
            for i, b in enumerate(self.output_meta):
                cuda.memcpy_dtoh_async(
                    self.host_outputs[i],
                    self.dev_ptrs_by_name[b["name"]],
                    self.stream
                )
                outs.append(self.host_outputs[i].copy() if copy_outputs else self.host_outputs[i])

            self.stream.synchronize()
            return outs

        else:
            # --- TRT 8/9: Legacy bindings API path ---
            # H2D: Copy to pinned buffer then to device
            np.copyto(self.host_input, batch_nchw, casting='no')
            cuda.memcpy_htod_async(
                self.dev_ptrs[self.input["index"]],
                self.host_input,
                self.stream
            )

            # Execute
            if self._exec_api == "execute_v2":
                # Synchronous execution requires prior operations to complete
                self.stream.synchronize()
                ok = self.context.execute_v2(self.dev_ptrs)
            else:
                # Async execution (default, no host sync)
                ok = self.context.execute_async_v2(self.dev_ptrs, self.stream.handle)
            
            if not ok:
                raise RuntimeError("TensorRT execution failed")

            # D2H: Copy outputs back to host
            outs: List[np.ndarray] = []
            for i, b in enumerate(self.output_meta):
                cuda.memcpy_dtoh_async(
                    self.host_outputs[i],
                    self.dev_ptrs[b["index"]],
                    self.stream
                )
                outs.append(self.host_outputs[i].copy() if copy_outputs else self.host_outputs[i])

            self.stream.synchronize()
            return outs

    def infer_with_timing(
        self,
        batch_nchw: np.ndarray,
        copy_outputs: bool = False
    ) -> Tuple[List[np.ndarray], Dict[str, float]]:
        """
        Run inference with detailed CUDA event timing.
        
        Returns:
            (outputs, timing_dict) where timing_dict contains:
                - h2d_ms: Host-to-device transfer time
                - compute_ms: Kernel execution time
                - d2h_ms: Device-to-host transfer time
                - total_ms: End-to-end time
        """
        batch_nchw = ensure_c_contig(batch_nchw, self.input["np_dtype"])
        if tuple(batch_nchw.shape) != tuple(self.input["shape"]):
            raise ValueError(
                f"Input shape {tuple(batch_nchw.shape)} != "
                f"engine expected {self.input['shape']}"
            )

        # Create CUDA events
        start = cuda.Event()
        h2d_done = cuda.Event()
        compute_done = cuda.Event()
        d2h_done = cuda.Event()

        if self._use_io_tensors:
            # --- TRT 10+: IO-tensor API path ---
            # H2D
            start.record(self.stream)
            np.copyto(self.host_input, batch_nchw, casting='no')
            cuda.memcpy_htod_async(
                self.dev_ptrs_by_name[self.input["name"]],
                self.host_input,
                self.stream
            )
            h2d_done.record(self.stream)

            # Execute (tensor addresses already set in _alloc_buffers)
            ok = self.context.execute_async_v3(self.stream.handle)
            if not ok:
                raise RuntimeError("TensorRT execute_async_v3 failed")
            
            compute_done.record(self.stream)

            # D2H
            outs: List[np.ndarray] = []
            for i, b in enumerate(self.output_meta):
                cuda.memcpy_dtoh_async(
                    self.host_outputs[i],
                    self.dev_ptrs_by_name[b["name"]],
                    self.stream
                )
                outs.append(self.host_outputs[i].copy() if copy_outputs else self.host_outputs[i])
            
            d2h_done.record(self.stream)
            d2h_done.synchronize()

        else:
            # --- TRT 8/9: Legacy bindings API path ---
            # H2D
            start.record(self.stream)
            np.copyto(self.host_input, batch_nchw, casting='no')
            cuda.memcpy_htod_async(
                self.dev_ptrs[self.input["index"]],
                self.host_input,
                self.stream
            )
            h2d_done.record(self.stream)

            # Execute
            if self._exec_api == "execute_v2":
                self.stream.synchronize()
                ok = self.context.execute_v2(self.dev_ptrs)
            else:
                ok = self.context.execute_async_v2(self.dev_ptrs, self.stream.handle)
            
            if not ok:
                raise RuntimeError("TensorRT execution failed")
            
            compute_done.record(self.stream)

            # D2H
            outs: List[np.ndarray] = []
            for i, b in enumerate(self.output_meta):
                cuda.memcpy_dtoh_async(
                    self.host_outputs[i],
                    self.dev_ptrs[b["index"]],
                    self.stream
                )
                outs.append(self.host_outputs[i].copy() if copy_outputs else self.host_outputs[i])
            
            d2h_done.record(self.stream)
            d2h_done.synchronize()

        # Compute elapsed times
        timings = {
            "h2d_ms": start.time_till(h2d_done),
            "compute_ms": h2d_done.time_till(compute_done),
            "d2h_ms": compute_done.time_till(d2h_done),
            "total_ms": start.time_till(d2h_done),
        }

        return outs, timings

    def benchmark(
        self,
        batch_nchw: np.ndarray,
        iters: int = 100,
        warmup: int = 20,
        validate_warmup: bool = False,
        cuda_events: bool = False
    ) -> Dict[str, float]:
        """
        Benchmark inference latency.
        
        Args:
            batch_nchw: Input batch
            iters: Number of timed iterations
            warmup: Number of warmup iterations
            validate_warmup: Check output consistency during warmup
            cuda_events: Use CUDA events for detailed timing breakdown
        
        Returns:
            Dictionary with timing statistics
        """
        # Warmup
        first_out = None
        for i in range(max(0, warmup)):
            out = self.infer(batch_nchw)
            if validate_warmup:
                if i == 0:
                    first_out = [o.copy() for o in out]
                else:
                    for j, (o0, o1) in enumerate(zip(first_out, out)):
                        if not np.allclose(o0, o1, rtol=1e-3, atol=1e-4):
                            print(f"[WARN] Warmup output {j} varies "
                                  "(may be normal for non-deterministic models)")

        # Timed runs
        if cuda_events:
            # Detailed GPU timing
            h2d_times, compute_times, d2h_times, total_times = [], [], [], []
            for _ in range(max(1, iters)):
                _, timings = self.infer_with_timing(batch_nchw)
                h2d_times.append(timings["h2d_ms"])
                compute_times.append(timings["compute_ms"])
                d2h_times.append(timings["d2h_ms"])
                total_times.append(timings["total_ms"])
            
            return {
                "iters": float(iters),
                "h2d_mean_ms": float(np.mean(h2d_times)),
                "compute_mean_ms": float(np.mean(compute_times)),
                "d2h_mean_ms": float(np.mean(d2h_times)),
                "total_mean_ms": float(np.mean(total_times)),
                "total_std_ms": float(np.std(total_times)),
                "total_min_ms": float(np.min(total_times)),
                "total_max_ms": float(np.max(total_times)),
                "fps": float(1000.0 / np.mean(total_times)) if np.mean(total_times) > 0 else float("inf"),
            }
        else:
            # Host-side timing (simpler, but includes some Python overhead)
            times = []
            for _ in range(max(1, iters)):
                t0 = time.perf_counter()
                _ = self.infer(batch_nchw)
                times.append(time.perf_counter() - t0)

            times = np.asarray(times, dtype=np.float64)
            return {
                "iters": float(iters),
                "mean_ms": float(times.mean() * 1000.0),
                "std_ms": float(times.std() * 1000.0),
                "min_ms": float(times.min() * 1000.0),
                "max_ms": float(times.max() * 1000.0),
                "fps": float(1.0 / times.mean()) if times.mean() > 0 else float("inf"),
            }


# ------------- Preprocessing --------------

def normalize(img_chw: np.ndarray, kind: str) -> np.ndarray:
    """
    Apply normalization to image.
    
    Args:
        img_chw: Image in CHW format, [0, 255] range
        kind: Normalization type ('none', 'unit', 'imagenet')
    
    Returns:
        Normalized image
    """
    if kind == "none":
        return img_chw
    img = img_chw / 255.0
    if kind == "unit":
        return img
    if kind == "imagenet":
        # ImageNet normalization: input is RGB CHW in [0,255]
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        return (img - mean) / std
    if kind in {"zero_center", "minus_one_to_one", "tanh"}:
        return (img - 0.5) / 0.5
    raise ValueError(f"Unknown normalization: {kind}")

def load_images_nchw(
    images_dir: str,
    expect_n: int,
    size_hw: Tuple[int, int],
    norm: str,
    dtype: np.dtype
) -> np.ndarray:
    """
    Load and preprocess images from directory.
    
    Args:
        images_dir: Directory containing images
        expect_n: Expected number of images (must match exactly)
        size_hw: Target size as (H, W)
        norm: Normalization type
        dtype: Output dtype (e.g., np.float16 for FP16 engines)
    
    Returns:
        Batch array of shape (N, 3, H, W)
    """
    if cv2 is None:
        raise RuntimeError(
            "OpenCV not available; install opencv-python or use --use-random."
        )
    
    H, W = size_hw
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    paths = [
        os.path.join(images_dir, f)
        for f in sorted(os.listdir(images_dir))
        if os.path.splitext(f.lower())[1] in exts
    ]
    
    if len(paths) != expect_n:
        raise RuntimeError(
            f"Found {len(paths)} images, need exactly {expect_n}. "
            f"Directory: {images_dir}"
        )

    batch = np.empty((expect_n, 3, H, W), dtype=np.float32)
    for i, p in enumerate(paths):
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            raise RuntimeError(f"Failed to read image: {p}")
        
        # BGR -> RGB
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        
        # Resize
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
        
        # HWC -> CHW
        im = np.transpose(im, (2, 0, 1))
        
        # Normalize
        im = normalize(im, norm)
        batch[i] = im

    # Convert to engine dtype (FP16 for FP16 engines)
    return batch.astype(dtype, copy=False)


# ----------------- CLI -------------------

def main():
    ap = argparse.ArgumentParser(
        description="Optimized single-stream TensorRT inference for RTX 5090",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pure model benchmark (no I/O overhead)
  %(prog)s --engine model.engine --use-random --iters 200 --cuda-events
  
  # Real images with async execution
  %(prog)s --engine model.engine --images-dir data/cams8 --save-outputs out.npz
  
  # Check if real-time capable (target: <33.3ms for 30 FPS)
  %(prog)s --engine model.engine --use-random --iters 100
        """
    )
    
    # Required
    ap.add_argument("--engine", required=True, help="Path to .engine file")
    
    # Input source
    ap.add_argument("--images-dir", help="Folder with exactly N images (N = engine batch)")
    ap.add_argument("--use-random", action="store_true",
                    help="Use random input instead of images (pure model test)")
    
    # Preprocessing
    ap.add_argument(
        "--norm",
        choices=["none", "unit", "imagenet", "zero_center", "minus_one_to_one", "tanh"],
        default="unit",
        help="Normalization for image inputs (default: unit)",
    )
    
    # Benchmarking
    ap.add_argument("--warmup", type=int, default=20,
                    help="Warmup iterations (default: 20)")
    ap.add_argument("--iters", type=int, default=100,
                    help="Timed iterations (default: 100)")
    ap.add_argument("--validate-warmup", action="store_true",
                    help="Check warmup output consistency")
    ap.add_argument("--cuda-events", action="store_true",
                    help="Use CUDA events for detailed timing (H2D/compute/D2H)")
    
    # Output
    ap.add_argument("--save-outputs", help="Save outputs to NPZ file")
    ap.add_argument("--copy-outputs", action="store_true",
                    help="Deep copy outputs (safe for storage, slower)")
    
    # Advanced
    ap.add_argument("--force-sync", action="store_true",
                    help="Force synchronous execution (not recommended)")
    ap.add_argument("--device", type=int, default=0,
                    help="CUDA device ID (default: 0)")
    ap.add_argument("--verbose", action="store_true",
                    help="Print detailed info")
    
    args = ap.parse_args()

    # Create runner
    with SimpleTrtRunner(
        args.engine,
        verbose=args.verbose,
        force_sync=args.force_sync,
        device=args.device
    ) as runner:
        
        # Get input specs
        N, C, H, W = runner.input["shape"]
        if C != 3:
            print(f"[WARN] Engine input channels = {C}, expected 3 (RGB). Proceeding.")

        # Build input batch
        if args.use_random:
            print(f"[INFO] Using random input: shape={runner.input['shape']}, "
                  f"dtype={runner.input['np_dtype']}")
            batch = np.random.random((N, C, H, W)).astype(np.float32)
            batch = batch.astype(runner.input["np_dtype"], copy=False)
        else:
            if not args.images_dir:
                print("[ERROR] Provide --images-dir or use --use-random.", file=sys.stderr)
                sys.exit(2)
            print(f"[INFO] Loading {N} images from {args.images_dir}")
            batch = load_images_nchw(
                args.images_dir, N, (H, W), args.norm, runner.input["np_dtype"]
            )

        # Run one inference to inspect outputs
        try:
            outs = runner.infer(batch, copy_outputs=args.copy_outputs)
        except Exception as e:
            print(f"[FATAL] Inference failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(3)

        print("\n[INFO] Output diagnostics (first run):")
        for i, (bmeta, arr) in enumerate(zip(runner.output_meta, outs)):
            print(f"  out{i}: name={bmeta['name']:<32} shape={tuple(arr.shape)} "
                  f"dtype={arr.dtype} min={arr.min():.6g} max={arr.max():.6g} "
                  f"mean={arr.mean():.6g}")

        # Benchmark
        stats = runner.benchmark(
            batch,
            iters=args.iters,
            warmup=args.warmup,
            validate_warmup=args.validate_warmup,
            cuda_events=args.cuda_events
        )
        
        print("\n========== Performance Stats ==========")
        if args.cuda_events:
            # Detailed GPU timing
            print(f"{'iterations':>20s}: {stats['iters']:.0f}")
            print(f"{'H2D mean (ms)':>20s}: {stats['h2d_mean_ms']:.3f}")
            print(f"{'Compute mean (ms)':>20s}: {stats['compute_mean_ms']:.3f}")
            print(f"{'D2H mean (ms)':>20s}: {stats['d2h_mean_ms']:.3f}")
            print(f"{'Total mean (ms)':>20s}: {stats['total_mean_ms']:.3f}")
            print(f"{'Total std (ms)':>20s}: {stats['total_std_ms']:.3f}")
            print(f"{'Total min (ms)':>20s}: {stats['total_min_ms']:.3f}")
            print(f"{'Total max (ms)':>20s}: {stats['total_max_ms']:.3f}")
            print(f"{'FPS (batch)':>20s}: {stats['fps']:.2f}")
            
            # Real-time assessment
            mean_ms = stats['total_mean_ms']
            if mean_ms <= 16.7:
                print(f"\n✓ Real-time capable: ≥60 FPS ({mean_ms:.1f}ms < 16.7ms)")
            elif mean_ms <= 33.3:
                print(f"\n✓ Real-time capable: ≥30 FPS ({mean_ms:.1f}ms < 33.3ms)")
            else:
                print(f"\n✗ Below real-time: {1000/mean_ms:.1f} FPS ({mean_ms:.1f}ms > 33.3ms)")
                print("  Consider: --use-random to test pure model performance")
        else:
            # Host-side timing
            for k in ["iters", "mean_ms", "std_ms", "min_ms", "max_ms", "fps"]:
                v = stats[k]
                if isinstance(v, float):
                    print(f"{k:>20s}: {v:.3f}")
                else:
                    print(f"{k:>20s}: {v}")
            
            mean_ms = stats['mean_ms']
            if mean_ms <= 16.7:
                print(f"\n✓ Real-time capable: ≥60 FPS ({mean_ms:.1f}ms < 16.7ms)")
            elif mean_ms <= 33.3:
                print(f"\n✓ Real-time capable: ≥30 FPS ({mean_ms:.1f}ms < 33.3ms)")
            else:
                print(f"\n✗ Below real-time: {1000/mean_ms:.1f} FPS ({mean_ms:.1f}ms > 33.3ms)")

        # Camera-level FPS (if batch represents multiple cameras)
        if N > 1:
            fps_per_batch = stats.get('fps', 1000.0 / stats.get('mean_ms', 33.3))
            camera_fps = fps_per_batch * N
            print(f"\nCamera-level throughput: {camera_fps:.1f} camera-FPS ({N} cameras × {fps_per_batch:.1f} batch-FPS)")

        # Save outputs if requested
        if args.save_outputs:
            out_dict = {bmeta["name"]: arr for bmeta, arr in zip(runner.output_meta, outs)}
            np.savez(args.save_outputs, **out_dict)
            print(f"\n[INFO] Saved outputs to: {args.save_outputs}")


if __name__ == "__main__":
    main()
