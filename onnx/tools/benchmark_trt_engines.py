#!/usr/bin/env python3
"""
Benchmark TensorRT, ONNX Runtime, and PyTorch VGGT pipelines.

For each quantisation directory the script discovers ``*.engine`` files,
measures inference latency using the optimized runner from
``onnx/tools/trt_inference.py``, and summarises FPS rankings.  When
requested it also benchmarks raw ONNX graphs (via onnxruntime) and the
baseline HuggingFace/PyTorch VGGT model on the selected device, reporting
depth sanity checks (min/max/positivity) for every run.
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - OpenCV optional
    cv2 = None

if __package__ in {None, ""}:
    import sys

    THIS_DIR = Path(__file__).resolve().parent
    REPO_ROOT = THIS_DIR.parent.parent
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from onnx.tools.trt_inference import SimpleTrtRunner as _SimpleTrtRunner


PRECISION_SUFFIX = {
    "fp32": "",
    "fp16": "_fp16",
    "bf16": "_bf16",
    "fp8": "_fp8",
    "int8": "_int8",
}


_IMAGE_CACHE: Dict[Tuple[str, int, int, int, str, bool], np.ndarray] = {}


def _collect_image_paths(images_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [
        path
        for path in sorted(images_dir.iterdir())
        if path.suffix.lower() in exts
    ]


def normalize(img_chw: np.ndarray, kind: str) -> np.ndarray:
    """Apply normalization to image tensors in CHW format."""
    if kind == "none":
        return img_chw
    img = img_chw / 255.0
    if kind == "unit":
        return img
    if kind == "imagenet":
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        return (img - mean) / std
    if kind in {"zero_center", "minus_one_to_one", "tanh"}:
        return (img - 0.5) / 0.5
    raise ValueError(f"Unknown normalization kind: {kind}")


def ensure_c_contig(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Ensure an array matches dtype and is C-contiguous."""
    arr = array.astype(dtype, copy=False)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    return arr


def load_images_nchw(
    images_dir: str,
    expect_n: int,
    size_hw: Tuple[int, int],
    norm: str,
    dtype: np.dtype,
) -> np.ndarray:
    """Load exactly N images from a directory into an NCHW tensor."""
    if cv2 is None:
        raise RuntimeError(
            "OpenCV not available; install opencv-python or use --use-random."
        )
    images_path = Path(images_dir)
    H, W = size_hw
    paths = _collect_image_paths(images_path)
    if len(paths) < expect_n:
        raise RuntimeError(
            f"Found {len(paths)} images, need at least {expect_n}. Directory: {images_dir}"
        )
    if len(paths) > expect_n:
        print(
            f"[INFO] Using the first {expect_n} images out of {len(paths)} available in {images_dir}."
        )
    selected = paths[:expect_n]
    batch = np.empty((expect_n, 3, H, W), dtype=np.float32)
    for idx, path in enumerate(selected):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {path}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
        chw = np.transpose(resized, (2, 0, 1))
        batch[idx] = normalize(chw, norm)
    return batch.astype(dtype, copy=False)


_TRT_RUNNER_CLS: Optional[Any] = None
_TRT_IMPORT_ERROR: Optional[Exception] = None


def _get_trt_runner_cls():
    """Lazily import SimpleTrtRunner to avoid hard dependency at startup."""
    global _TRT_RUNNER_CLS, _TRT_IMPORT_ERROR
    if _TRT_RUNNER_CLS is not None:
        return _TRT_RUNNER_CLS
    if _TRT_IMPORT_ERROR is not None:
        raise _TRT_IMPORT_ERROR
    try:
        module = importlib.import_module("onnx.tools.trt_inference")
    except SystemExit as exc:
        _TRT_IMPORT_ERROR = RuntimeError(
            "TensorRT/PyCUDA dependencies unavailable; install them to benchmark TensorRT engines."
        )
        raise _TRT_IMPORT_ERROR from exc
    except Exception as exc:  # pragma: no cover - import failure
        _TRT_IMPORT_ERROR = exc
        raise
    _TRT_RUNNER_CLS = module.SimpleTrtRunner  # type: ignore[attr-defined]
    return _TRT_RUNNER_CLS


def _cache_key(
    images_dir: Path,
    count: int,
    size_hw: Tuple[int, int],
    norm: str,
    exact: bool,
) -> Tuple[str, int, int, int, str, bool]:
    return (
        str(images_dir.resolve()),
        int(count),
        int(size_hw[0]),
        int(size_hw[1]),
        norm,
        exact,
    )


def _load_images_subset(
    images_dir: Path,
    count: int,
    size_hw: Tuple[int, int],
    norm: str,
    *,
    exact: bool,
    expect_n: Optional[int] = None,
) -> np.ndarray:
    """Load ``count`` images, optionally enforcing exact cardinality."""
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is required for --images-dir inputs. Install with `pip install opencv-python`."
        )
    H, W = size_hw
    paths = _collect_image_paths(images_dir)
    if exact and expect_n is not None and len(paths) != expect_n:
        raise RuntimeError(
            f"Found {len(paths)} images but need exactly {expect_n}. Directory: {images_dir}"
        )
    if len(paths) < count:
        raise RuntimeError(
            f"Found {len(paths)} images but need at least {count}. Directory: {images_dir}"
        )
    selected = paths[:count]
    batch = np.empty((count, 3, H, W), dtype=np.float32)
    for idx, path in enumerate(selected):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read image: {path}")
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
        chw = np.transpose(resized, (2, 0, 1))
        batch[idx] = normalize(chw, norm)
    return batch


def _get_image_batch(
    images_dir: Path,
    count: int,
    size_hw: Tuple[int, int],
    norm: str,
    *,
    exact: bool,
    dtype: np.dtype,
    expect_n: Optional[int] = None,
) -> np.ndarray:
    """Load (and cache) image batches for reuse across backends."""
    key = _cache_key(images_dir, count, size_hw, norm, exact)
    if key not in _IMAGE_CACHE:
        if exact:
            _IMAGE_CACHE[key] = load_images_nchw(
                str(images_dir),
                count,
                size_hw=size_hw,
                norm=norm,
                dtype=np.float32,
            )
        else:
            _IMAGE_CACHE[key] = _load_images_subset(
                images_dir,
                count,
                size_hw,
                norm,
                exact=False,
                expect_n=expect_n,
            )
    batch = _IMAGE_CACHE[key]
    return batch.astype(dtype, copy=True)


def detect_precision(engine_path: Path) -> str:
    """Best-effort precision classification from filename suffix."""
    name = engine_path.name
    # Check longer suffixes first to avoid matching the empty fp32 suffix prematurely.
    sorted_suffixes = sorted(
        PRECISION_SUFFIX.items(),
        key=lambda kv: len(kv[1]),
        reverse=True,
    )
    for prec, suffix in sorted_suffixes:
        if suffix:
            token = f"{suffix}.engine"
            if name.endswith(token):
                return prec
    # Fallback to fp32 (no suffix match)
    return "fp32"


def prepare_batch(
    runner: "_SimpleTrtRunner",
    *,
    images_dir: Optional[Path],
    use_random: bool,
    norm: str,
    engine_path: Path,
) -> np.ndarray:
    """Prepare an input batch matching the engine's expectations."""
    shape = tuple(runner.input["shape"])
    dtype = runner.input["np_dtype"]
    expect_n = shape[0]

    if not use_random and images_dir is not None:
        try:
            batch = _get_image_batch(
                images_dir,
                expect_n,
                size_hw=(shape[2], shape[3]),
                norm=norm,
                exact=True,
                dtype=np.float32,
                expect_n=expect_n,
            )
        except Exception as exc:
            print(f"[WARN] Falling back to random input for {engine_path.name}: {exc}")
            batch = np.random.randn(*shape).astype(np.float32)
    else:
        batch = np.random.randn(*shape).astype(np.float32)

    if batch.dtype != dtype:
        batch = batch.astype(dtype, copy=False)

    return ensure_c_contig(batch, dtype)


def _iter_depth_maps(
    arrays: Sequence[np.ndarray],
    names: Sequence[str],
) -> Iterable[Tuple[str, np.ndarray]]:
    """Yield candidate depth maps (name, array) from model outputs."""
    for idx, array in enumerate(arrays):
        name = names[idx] if idx < len(names) else f"output_{idx}"
        arr = np.asarray(array)
        if arr.ndim >= 5:
            arr = arr.reshape(arr.shape[0] * arr.shape[1], *arr.shape[2:])
        if arr.ndim == 4:
            if arr.shape[1] in {1, 2}:
                arr = arr[:, 0, :, :]
            elif arr.shape[-1] in {1, 2}:
                arr = arr[:, :, :, 0]
            elif arr.shape[1] == 3 or arr.shape[-1] == 3:
                continue
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim != 3:
            continue
        H, W = arr.shape[-2:]
        if min(H, W) < 32:
            continue
        for view_idx in range(arr.shape[0]):
            depth = np.asarray(arr[view_idx], dtype=np.float32)
            yield f"{name}[{view_idx}]", depth


def _log_depth_candidates(
    candidates: Sequence[Tuple[str, np.ndarray]],
) -> Dict[str, Optional[float]]:
    """Print per-view depth statistics and aggregate summary."""
    if not candidates:
        print("    depth: [warn] no plausible depth outputs detected.")
        return {
            "min": None,
            "max": None,
            "positive_fraction": None,
            "positive": 0,
            "finite": 0,
            "samples": 0,
        }

    global_min: Optional[float] = None
    global_max: Optional[float] = None
    total_finite = 0
    total_positive = 0

    for label, depth in candidates:
        arr = np.asarray(depth, dtype=np.float32)
        finite = np.isfinite(arr)
        finite_count = int(finite.sum())
        total = arr.size
        positive = int(np.count_nonzero(finite & (arr > 0)))
        total_finite += finite_count
        total_positive += positive

        if finite_count:
            dmin = float(np.min(arr[finite]))
            dmax = float(np.max(arr[finite]))
            global_min = dmin if global_min is None else min(global_min, dmin)
            global_max = dmax if global_max is None else max(global_max, dmax)
        else:
            dmin = float("nan")
            dmax = float("nan")

        pos_pct = (positive / finite_count * 100.0) if finite_count else 0.0
        print(
            f"    depth[{label}]: min={dmin:8.4f}  max={dmax:8.4f}  "
            f"pos={positive}/{total} ({pos_pct:5.1f}% finite)"
        )

    positive_fraction = (
        total_positive / total_finite if total_finite > 0 else None
    )
    return {
        "min": global_min,
        "max": global_max,
        "positive_fraction": positive_fraction,
        "positive": total_positive,
        "finite": total_finite,
        "samples": len(candidates),
    }


def _format_depth_summary(depth_stats: Dict[str, Optional[float]]) -> str:
    if not depth_stats or depth_stats.get("min") is None or depth_stats.get("max") is None:
        return "depth[min=-- max=-- pos=--]"
    min_val = depth_stats["min"]
    max_val = depth_stats["max"]
    frac = depth_stats.get("positive_fraction")
    if frac is None:
        return f"depth[min={min_val:0.3f} max={max_val:0.3f} pos=--]"
    return f"depth[min={min_val:0.3f} max={max_val:0.3f} pos={frac*100:0.1f}%]"


def _prepare_generic_batch(
    *,
    images_dir: Optional[Path],
    num_views: int,
    size_hw: Tuple[int, int],
    norm: str,
    use_random: bool,
    dtype: np.dtype,
) -> np.ndarray:
    if not use_random and images_dir is not None:
        try:
            return _get_image_batch(
                images_dir,
                num_views,
                size_hw=size_hw,
                norm=norm,
                exact=False,
                dtype=np.float32,
            ).astype(dtype, copy=False)
        except Exception as exc:
            print(f"[WARN] Could not load images from {images_dir}: {exc}. Using random input instead.")
    return np.random.randn(num_views, 3, size_hw[0], size_hw[1]).astype(dtype, copy=False)


def benchmark_huggingface_model(
    *,
    model_id: Optional[str],
    weights_path: Optional[Path],
    device: str,
    iters: int,
    warmup: int,
    images_dir: Optional[Path],
    use_random: bool,
    norm: str,
    num_views: Optional[int],
    image_size: int,
    autocast: bool,
) -> Dict[str, object]:
    """Benchmark the baseline PyTorch VGGT model."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for HuggingFace benchmarking.") from exc

    from vggt.models.vggt import VGGT  # Local import to avoid heavy dependency unless needed

    if weights_path is None and not model_id:
        raise ValueError("Provide either --hf-weights or --hf-model-id to benchmark the PyTorch model.")

    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but PyTorch does not detect CUDA.")

    if weights_path is not None:
        model = VGGT()
        state_dict = torch.load(str(weights_path), map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        variant = weights_path.name
    else:
        model = VGGT.from_pretrained(model_id)  # type: ignore[arg-type]
        variant = model_id or "VGGT"

    model.to(device_obj)
    model.eval()

    available_images = None
    if images_dir is not None:
        available_images = len(_collect_image_paths(images_dir))
    effective_views = num_views or available_images or 8
    if effective_views <= 0:
        raise ValueError("Could not determine number of views to benchmark. Use --num-views.")

    batch_np = _prepare_generic_batch(
        images_dir=images_dir,
        num_views=effective_views,
        size_hw=(image_size, image_size),
        norm=norm,
        use_random=use_random,
        dtype=np.float32,
    )
    batch_tensor = torch.from_numpy(batch_np).to(device_obj)

    autocast_dtype = None
    if autocast and device_obj.type == "cuda":
        major, _minor = torch.cuda.get_device_capability(device_obj)
        autocast_dtype = torch.bfloat16 if major >= 8 else torch.float16

    def forward_once() -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            if autocast_dtype is not None:
                with torch.cuda.amp.autocast(dtype=autocast_dtype):
                    outputs = model(batch_tensor)
            else:
                outputs = model(batch_tensor)
        return outputs

    # Warmup
    for _ in range(max(0, warmup)):
        _ = forward_once()
        if device_obj.type == "cuda":
            torch.cuda.synchronize(device_obj)

    timings_ms: List[float] = []
    outputs_last: Optional[Dict[str, torch.Tensor]] = None
    for _ in range(iters):
        start = time.perf_counter()
        outputs_last = forward_once()
        if device_obj.type == "cuda":
            torch.cuda.synchronize(device_obj)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timings_ms.append(float(elapsed_ms))

    mean_ms = float(np.mean(timings_ms)) if timings_ms else float("nan")
    std_ms = float(np.std(timings_ms)) if timings_ms else float("nan")
    fps = 1000.0 / mean_ms if mean_ms > 0 else float("inf")

    depth_stats = {
        "min": None,
        "max": None,
        "positive_fraction": None,
        "positive": 0,
        "finite": 0,
        "samples": 0,
    }
    if outputs_last is not None and "depth" in outputs_last:
        depth_np = outputs_last["depth"].detach().cpu().numpy()
        candidates = list(_iter_depth_maps([depth_np], ["depth"]))
        depth_stats = _log_depth_candidates(candidates)

    precision = "autocast" if autocast_dtype is not None else "fp32"
    return {
        "backend": "huggingface",
        "variant": variant,
        "precision": precision,
        "model": variant,
        "fps": fps,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "depth": depth_stats,
        "device": str(device_obj),
        "iters": iters,
    }


def _ort_type_to_dtype(type_str: str) -> np.dtype:
    mapping = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(bfloat16)": np.dtype("bfloat16"),
        "tensor(double)": np.float64,
        "tensor(int32)": np.int32,
        "tensor(int64)": np.int64,
    }
    dtype = mapping.get(type_str)
    if dtype is None:
        raise ValueError(f"Unsupported ONNX input dtype: {type_str}")
    return dtype


def _resolve_dim(value: object, fallback: Optional[int], *, name: str) -> int:
    if isinstance(value, (int, np.integer)):
        return int(value)
    if fallback is not None:
        return int(fallback)
    raise ValueError(f"Dimension '{name}' is dynamic; specify --num-views/--image-size to resolve it.")


def benchmark_onnx_model(
    model_path: Path,
    *,
    iters: int,
    warmup: int,
    images_dir: Optional[Path],
    use_random: bool,
    norm: str,
    num_views: Optional[int],
    image_size: int,
    provider: Optional[str],
) -> Dict[str, object]:
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("onnxruntime is required for ONNX benchmarking.") from exc

    available_providers = ort.get_available_providers()
    if provider is not None:
        providers = [provider]
    elif "CUDAExecutionProvider" in available_providers:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(str(model_path), providers=providers)
    inputs = session.get_inputs()
    if not inputs:
        raise RuntimeError(f"ONNX model has no inputs: {model_path}")
    input_meta = inputs[0]
    input_name = input_meta.name
    input_shape = tuple(input_meta.shape)
    input_dtype = _ort_type_to_dtype(input_meta.type)

    available_images = None
    if images_dir is not None:
        available_images = len(_collect_image_paths(images_dir))

    height_fallback = image_size
    width_fallback = image_size
    views_fallback = num_views or available_images or 8

    rank = len(input_shape)
    if rank == 4:
        # [V, C, H, W]
        views = _resolve_dim(input_shape[0], views_fallback, name="views")
        channels = _resolve_dim(input_shape[1], 3, name="channels")
        height = _resolve_dim(input_shape[2], height_fallback, name="height")
        width = _resolve_dim(input_shape[3], width_fallback, name="width")
        if channels != 3:
            raise ValueError(f"Expected 3 channels but ONNX model requires {channels}.")
        batch_np = _prepare_generic_batch(
            images_dir=images_dir,
            num_views=views,
            size_hw=(height, width),
            norm=norm,
            use_random=use_random,
            dtype=np.float32,
        )
        input_array = batch_np.astype(input_dtype, copy=False)
    elif rank == 5:
        # [B, V, C, H, W]
        batch = _resolve_dim(input_shape[0], 1, name="batch")
        views = _resolve_dim(input_shape[1], views_fallback, name="views")
        channels = _resolve_dim(input_shape[2], 3, name="channels")
        height = _resolve_dim(input_shape[3], height_fallback, name="height")
        width = _resolve_dim(input_shape[4], width_fallback, name="width")
        if channels != 3:
            raise ValueError(f"Expected 3 channels but ONNX model requires {channels}.")
        batch_np = _prepare_generic_batch(
            images_dir=images_dir,
            num_views=views,
            size_hw=(height, width),
            norm=norm,
            use_random=use_random,
            dtype=np.float32,
        )
        batch_np = batch_np.astype(input_dtype, copy=False)
        input_array = batch_np.reshape(1, views, 3, height, width)
        if batch != 1:
            input_array = np.repeat(input_array, batch, axis=0)
    else:
        raise ValueError(
            f"Unsupported ONNX input rank {rank} for {model_path}. Expected 4D or 5D tensors."
        )

    input_array = np.ascontiguousarray(input_array)

    # Warmup
    for _ in range(max(0, warmup)):
        session.run(None, {input_name: input_array})

    timings_ms: List[float] = []
    outputs_last: Optional[List[np.ndarray]] = None
    for _ in range(iters):
        start = time.perf_counter()
        outputs_last = session.run(None, {input_name: input_array})
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timings_ms.append(float(elapsed_ms))

    mean_ms = float(np.mean(timings_ms)) if timings_ms else float("nan")
    std_ms = float(np.std(timings_ms)) if timings_ms else float("nan")
    fps = 1000.0 / mean_ms if mean_ms > 0 else float("inf")

    depth_stats = {
        "min": None,
        "max": None,
        "positive_fraction": None,
        "positive": 0,
        "finite": 0,
        "samples": 0,
    }
    if outputs_last is not None:
        output_names = [out.name or f"output_{idx}" for idx, out in enumerate(session.get_outputs())]
        candidates = list(_iter_depth_maps(outputs_last, output_names))
        depth_stats = _log_depth_candidates(candidates)

    variant = model_path.parent.name or model_path.stem
    return {
        "backend": "onnxruntime",
        "variant": variant,
        "precision": input_meta.type,
        "model": str(model_path),
        "provider": providers[0] if providers else None,
        "fps": fps,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "depth": depth_stats,
        "iters": iters,
    }


def benchmark_engine(
    engine_path: Path,
    *,
    iters: int,
    warmup: int,
    images_dir: Optional[Path],
    use_random: bool,
    norm: str,
    cuda_events: bool,
    verbose_runner: bool,
) -> Dict[str, object]:
    """Run the latency benchmark for a single engine."""
    runner_cls = _get_trt_runner_cls()
    with runner_cls(
        str(engine_path),
        verbose=verbose_runner,
    ) as runner:
        batch = prepare_batch(
            runner,
            images_dir=images_dir,
            use_random=use_random,
            norm=norm,
            engine_path=engine_path,
        )
        outputs_sample = runner.infer(batch, copy_outputs=True)
        output_names = [
            meta.get("name", f"output_{idx}") for idx, meta in enumerate(runner.output_meta)
        ]
        depth_candidates = list(_iter_depth_maps(outputs_sample, output_names))
        depth_stats = _log_depth_candidates(depth_candidates)
        stats = runner.benchmark(
            batch,
            iters=iters,
            warmup=warmup,
            cuda_events=cuda_events,
        )

    # Normalise stat keys between CUDA-event and host timing paths.
    if cuda_events:
        latency_ms = float(stats["total_mean_ms"])
        jitter_ms = float(stats["total_std_ms"])
    else:
        latency_ms = float(stats["mean_ms"])
        jitter_ms = float(stats["std_ms"])

    fps = 1000.0 / latency_ms if latency_ms > 0 else float("inf")
    return {
        "mean_ms": latency_ms,
        "std_ms": jitter_ms,
        "fps": fps,
        "depth": depth_stats,
    }


def summarize(results: List[Dict[str, object]]) -> None:
    """Pretty-print the collected benchmark results ordered by FPS."""
    if not results:
        print("[INFO] No benchmarks executed.")
        return

    results_sorted = sorted(results, key=lambda r: r.get("fps", 0.0), reverse=True)

    print("\n=== Benchmark Summary ===")
    header = (
        f"{'Rank':>4}  {'Backend':>10}  {'Variant':>18}  {'Precision':>10}  "
        f"{'FPS':>8}  {'Latency(ms)':>12}  {'DepthMin':>10}  {'DepthMax':>10}  Detail"
    )
    print(header)
    print("-" * len(header))

    for idx, row in enumerate(results_sorted, start=1):
        backend = str(row.get("backend", "?"))
        variant = str(row.get("variant") or row.get("quant_mode") or "?")
        precision = str(row.get("precision", "--"))
        fps = float(row.get("fps", float("nan")))
        mean_ms = float(row.get("mean_ms", float("nan")))
        depth_stats = row.get("depth") or {}
        depth_min = depth_stats.get("min")
        depth_max = depth_stats.get("max")
        depth_min_str = f"{depth_min:10.3f}" if depth_min is not None else f"{'--':>10}"
        depth_max_str = f"{depth_max:10.3f}" if depth_max is not None else f"{'--':>10}"
        detail = row.get("engine") or row.get("model") or ""
        print(
            f"{idx:4d}  {backend:>10}  {variant:>18}  {precision:>10}  "
            f"{fps:8.2f}  {mean_ms:12.2f}  {depth_min_str}  {depth_max_str}  {detail}"
        )

    best = results_sorted[0]
    print("\nFastest configuration:")
    print(
        f"  {best.get('backend', '?')} / {best.get('variant', '?')} "
        f"→ {best.get('fps', float('nan')):.2f} FPS ({best.get('mean_ms', float('nan')):.2f} ms)"
    )

    over_30 = [r for r in results_sorted if r.get("fps", 0.0) >= 30.0]
    over_15 = [r for r in results_sorted if r.get("fps", 0.0) >= 15.0]
    print(f"\n≥30 FPS configurations: {len(over_30)}")
    print(f"≥15 FPS configurations: {len(over_15)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark TensorRT engines under onnx_exports/<quant_mode>.",
    )
    parser.add_argument(
        "--root",
        default="onnx_exports",
        help="Root directory containing quantisation subdirectories.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Timed iterations per engine (default: 100).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Warmup iterations per engine (default: 20).",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        help="Optional image directory with exactly N frames (fallbacks to random).",
    )
    parser.add_argument(
        "--use-random",
        action="store_true",
        help="Force random input even if --images-dir is supplied.",
    )
    parser.add_argument(
        "--norm",
        choices=["none", "unit", "imagenet", "zero_center", "minus_one_to_one", "tanh"],
        default="unit",
        help="Normalization to apply when loading images (default: unit).",
    )
    parser.add_argument(
        "--cuda-events",
        action="store_true",
        help="Use CUDA events for timing breakdown (default: host timing).",
    )
    parser.add_argument(
        "--verbose-runner",
        action="store_true",
        help="Enable verbose TensorRT runner logs.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Optional path to save raw results as JSON.",
    )
    parser.add_argument(
        "--num-views",
        type=int,
        help="Number of camera views for ONNX/PyTorch benchmarks when model shapes are dynamic.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=518,
        help="Square image size for dynamic ONNX/PyTorch inputs (default: 518).",
    )
    parser.add_argument(
        "--hf-weights",
        type=Path,
        help="Path to VGGT state_dict (.pt) for PyTorch benchmarking.",
    )
    parser.add_argument(
        "--hf-model-id",
        help="HuggingFace model identifier (used when --hf-weights is omitted).",
    )
    parser.add_argument(
        "--hf-device",
        default="cuda",
        help="Device for PyTorch benchmarking (default: cuda).",
    )
    parser.add_argument(
        "--hf-no-autocast",
        action="store_true",
        help="Disable CUDA autocast during PyTorch benchmarking (defaults to enabled when supported).",
    )
    parser.add_argument(
        "--onnx-models",
        nargs="*",
        type=Path,
        help="Explicit ONNX model files to benchmark (in addition to auto-discovery).",
    )
    parser.add_argument(
        "--onnx-glob",
        type=str,
        help="Glob relative to --root for discovering ONNX models (e.g. '*/vggt-*.onnx').",
    )
    parser.add_argument(
        "--onnx-provider",
        type=str,
        help="Preferred ONNX Runtime execution provider (default: CUDA when available).",
    )
    parser.add_argument(
        "--trt-engines",
        nargs="*",
        type=Path,
        help="Explicit TensorRT engine files to benchmark (skips discovery when provided).",
    )

    args = parser.parse_args()
    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"[ERROR] Root directory does not exist: {root}")

    results: List[Dict[str, object]] = []
    seen_trt: set[str] = set()

    if not args.trt_engines:
        quant_dirs = sorted([p for p in root.iterdir() if p.is_dir()])

        for quant_dir in quant_dirs:
            quant_mode = quant_dir.name
            engine_paths = sorted(quant_dir.glob("*.engine"))
            if not engine_paths:
                continue

            print(f"\n[INFO] Benchmarking quantisation mode '{quant_mode}'")
            for engine_path in engine_paths:
                precision = detect_precision(engine_path)
                try:
                    stats = benchmark_engine(
                        engine_path,
                        iters=args.iters,
                        warmup=args.warmup,
                        images_dir=args.images_dir,
                        use_random=args.use_random,
                        norm=args.norm,
                        cuda_events=args.cuda_events,
                        verbose_runner=args.verbose_runner,
                    )
                except Exception as exc:
                    print(f"[ERROR] Benchmark failed for {engine_path}: {exc}")
                    if _TRT_IMPORT_ERROR is not None:
                        break
                    continue

                record: Dict[str, object] = {
                    "backend": "tensorrt",
                    "variant": quant_mode,
                    "precision": precision,
                    "engine": str(engine_path),
                    "model": str(engine_path),
                    "fps": stats["fps"],
                    "mean_ms": stats["mean_ms"],
                    "std_ms": stats["std_ms"],
                    "depth": stats.get("depth"),
                }
                results.append(record)
                seen_trt.add(str(engine_path.resolve()))
                depth_summary = _format_depth_summary(record.get("depth") or {})
                print(
                    f"  TRT[{precision:>5}] → {stats['fps']:.2f} FPS "
                    f"({stats['mean_ms']:.2f} ± {stats['std_ms']:.2f} ms)  {depth_summary}  [{engine_path.name}]"
                )

            if _TRT_IMPORT_ERROR is not None:
                break

    if args.trt_engines:
        print("\n[INFO] Benchmarking explicit TensorRT engines")
        for engine_path in args.trt_engines:
            engine_resolved = engine_path.expanduser().resolve()
            if not engine_resolved.exists():
                print(f"[WARN] TensorRT engine not found: {engine_resolved}")
                continue
            if str(engine_resolved) in seen_trt:
                continue
            precision = detect_precision(engine_resolved)
            try:
                stats = benchmark_engine(
                    engine_resolved,
                    iters=args.iters,
                    warmup=args.warmup,
                    images_dir=args.images_dir,
                    use_random=args.use_random,
                    norm=args.norm,
                    cuda_events=args.cuda_events,
                    verbose_runner=args.verbose_runner,
                )
            except Exception as exc:
                print(f"[ERROR] Benchmark failed for {engine_resolved}: {exc}")
                continue

            record = {
                "backend": "tensorrt",
                "variant": engine_resolved.parent.name or "custom",
                "precision": precision,
                "engine": str(engine_resolved),
                "model": str(engine_resolved),
                "fps": stats["fps"],
                "mean_ms": stats["mean_ms"],
                "std_ms": stats["std_ms"],
                "depth": stats.get("depth"),
            }
            results.append(record)
            depth_summary = _format_depth_summary(record.get("depth") or {})
            print(
                f"  TRT[{precision:>5}] → {stats['fps']:.2f} FPS "
                f"({stats['mean_ms']:.2f} ± {stats['std_ms']:.2f} ms)  {depth_summary}  [{engine_resolved.name}]"
            )

    hf_requested = args.hf_weights is not None or args.hf_model_id
    if hf_requested:
        print("\n[INFO] Benchmarking PyTorch VGGT model")
        try:
            hf_stats = benchmark_huggingface_model(
                model_id=args.hf_model_id,
                weights_path=args.hf_weights,
                device=args.hf_device,
                iters=args.iters,
                warmup=args.warmup,
                images_dir=args.images_dir,
                use_random=args.use_random,
                norm=args.norm,
                num_views=args.num_views,
                image_size=args.image_size,
                autocast=not args.hf_no_autocast,
            )
        except Exception as exc:
            print(f"[ERROR] HuggingFace benchmark failed: {exc}")
        else:
            results.append(hf_stats)
            depth_summary = _format_depth_summary(hf_stats.get("depth") or {})
            device = hf_stats.get("device", args.hf_device)
            print(
                f"  HF[{hf_stats.get('precision', 'fp32')}] ({device}) → {hf_stats['fps']:.2f} FPS "
                f"({hf_stats['mean_ms']:.2f} ± {hf_stats['std_ms']:.2f} ms)  {depth_summary}"
            )

    onnx_candidates: Dict[Path, None] = {}

    def _add_onnx_candidate(path_like: Path) -> None:
        resolved = path_like.expanduser().resolve()
        if resolved.suffix.lower() != ".onnx":
            return
        if not resolved.exists():
            print(f"[WARN] ONNX model not found: {resolved}")
            return
        onnx_candidates[resolved] = None

    if args.onnx_models:
        for candidate in args.onnx_models:
            _add_onnx_candidate(candidate)

    if args.onnx_glob:
        for candidate in root.glob(args.onnx_glob):
            _add_onnx_candidate(candidate)

    if not onnx_candidates:
        for candidate in root.rglob("*.onnx"):
            _add_onnx_candidate(candidate)

    for model_path in sorted(onnx_candidates):
        try:
            display_path = model_path.relative_to(root)
        except ValueError:
            display_path = model_path
        print(f"\n[INFO] Benchmarking ONNX model '{display_path}'")
        try:
            ort_stats = benchmark_onnx_model(
                model_path,
                iters=args.iters,
                warmup=args.warmup,
                images_dir=args.images_dir,
                use_random=args.use_random,
                norm=args.norm,
                num_views=args.num_views,
                image_size=args.image_size,
                provider=args.onnx_provider,
            )
        except Exception as exc:
            print(f"[ERROR] ONNX benchmark failed for {model_path}: {exc}")
            continue

        results.append(ort_stats)
        depth_summary = _format_depth_summary(ort_stats.get("depth") or {})
        provider = ort_stats.get("provider", "auto")
        print(
            f"  ONNX[{provider}] → {ort_stats['fps']:.2f} FPS "
            f"({ort_stats['mean_ms']:.2f} ± {ort_stats['std_ms']:.2f} ms)  {depth_summary}"
        )

    summarize(results)

    if args.json:
        try:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            with args.json.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"\n[INFO] Saved benchmark results to {args.json}")
        except Exception as exc:
            print(f"[WARN] Failed to save JSON: {exc}")


if __name__ == "__main__":
    main()
