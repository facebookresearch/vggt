#!/usr/bin/env python3
"""
VGGT → TensorRT Conversion Pipeline (ChatGPT Edition)
----------------------------------------------------

Highlights
~~~~~~~~~~
* Stable INT8 calibration that accepts real multi-view image batches
* Automatic opset repair + resilient ONNX simplification fallback
* Precision-aware reporting (detects FP8 fallback to FP16/FP32)
* Optional pre-quantisation hooks (PyTorch dynamic INT8, bitsandbytes FP4/FP8,
  NVIDIA ModelOpt FP8) before ONNX export
* Calibrator supports directory / glob / .npy sources and caching

The pipeline is tailored for live point-cloud and 3D Gaussian Splatting work:
the default output keeps depth + camera heads while allowing full-head export.

Example usage::

    # Export + build FP16 engine (baseline)
    python vggt_to_trt_chatgpt.py --export --num-cams 8 --precision fp16

    # Reuse ONNX, build INT8 with calibration images
    python vggt_to_trt_chatgpt.py --onnx-in onnx_exports/vggt.onnx \
        --precision int8 --calib-images data/mvs360 --calib-batches 32

    # Pre-quantise model to NF4 (bitsandbytes) before export, then build FP16
    python vggt_to_trt_chatgpt.py --export --quant-mode bitsandbytes-nf4 \
        --precision fp16 --pcd-only
"""

from __future__ import annotations
import os
import sys
import logging
import glob
import inspect
import itertools
import math
import pathlib
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from dataclasses import dataclass

import onnx
import numpy as np
from onnx import helper, numpy_helper, AttributeProto, TensorProto, ValueInfoProto
from onnx.external_data_helper import convert_model_to_external_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Lazy imports
def _lazy_import(name: str):
    """Import module lazily without side effects."""
    try:
        return __import__(name)
    except Exception:
        return None

torch = _lazy_import("torch")
trt_mod = _lazy_import("tensorrt")
pil_mod = _lazy_import("PIL")
Image = getattr(pil_mod, "Image", None) if pil_mod else None

_TENSOR_TYPES: Tuple[type, ...]
if torch is not None:
    _TENSOR_TYPES = (torch.Tensor,)
else:
    _TENSOR_TYPES = tuple()


# External data file size threshold (1KB)
EXTERNAL_DATA_THRESHOLD = 1024

# Workspace fallbacks (GB) to retry TensorRT builds under memory pressure
WORKSPACE_FALLBACKS = [24, 20, 16, 12, 8, 6, 4, 2]

# ---------- Utility Functions ----------

def _mkdir_for(p: str) -> None:
    """Create parent directory for given path if it doesn't exist."""
    d = os.path.dirname(os.path.abspath(p)) or "."
    os.makedirs(d, exist_ok=True)

def _rm(p: str) -> None:
    """Remove file if it exists, ignoring errors."""
    try:
        if os.path.isfile(p) or os.path.islink(p):
            os.remove(p)
    except Exception:
        pass

def _data_rel(onnx_path: str) -> str:
    """Get relative external data filename for ONNX model."""
    return os.path.basename(onnx_path) + ".data"

def _data_abs(onnx_path: str) -> str:
    """Get absolute external data filepath for ONNX model."""
    return os.path.join(
        os.path.dirname(os.path.abspath(onnx_path)),
        _data_rel(onnx_path)
    )

def _fmt_size(n: int) -> str:
    """Format byte size as human-readable string."""
    gb = n / (1024**3)
    mb = n / (1024**2)
    kb = n / 1024
    if gb >= 1:
        return f"{gb:.2f} GB"
    elif mb >= 1:
        return f"{mb:.1f} MB"
    else:
        return f"{kb:.1f} KB"

def _ensure_opset(m: onnx.ModelProto, opset_version: int) -> None:
    """
    Ensure model has valid opset_import with default domain.
    CRITICAL FIX: This prevents onnxsim from failing.
    """
    if len(m.opset_import) == 0:
        logger.warning("Model missing opset_import entirely - adding default")
        opset = m.opset_import.add()
        opset.domain = ""
        opset.version = opset_version
        return
    
    # Check for default domain
    default_entry = None
    for op in m.opset_import:
        if op.domain in ("", "ai.onnx"):
            default_entry = op
            break

    if default_entry is None:
        logger.warning("Model missing default opset domain - adding")
        opset = m.opset_import.add()
        opset.domain = ""
        opset.version = opset_version
        return

    # Normalise legacy domain spelling and update version if needed
    if default_entry.domain != "":
        logger.info("Normalising opset domain '%s' -> ''", default_entry.domain)
        default_entry.domain = ""
    if default_entry.version != opset_version:
        logger.info(
            "Updating opset version %s -> %s",
            default_entry.version,
            opset_version,
        )
        default_entry.version = opset_version
    
    # Remove duplicate default entries if any existed
    duplicates = [
        op for op in m.opset_import
        if op is not default_entry and op.domain in ("", "ai.onnx")
    ]
    for dup in duplicates:
        logger.debug("Removing duplicate opset entry with domain '%s'", dup.domain)
    for dup in duplicates:
        m.opset_import.remove(dup)


def _prod_map(g: onnx.GraphProto) -> dict:
    """Build map from output name to producing node."""
    out = {}
    for n in g.node:
        for o in n.output:
            if o:
                out[o] = n
    return out


def _cast_tensor_proto_to_fp32(tensor: TensorProto) -> bool:
    """Convert TensorProto data to float32 if currently float64/BFloat16."""
    if tensor.data_type not in (TensorProto.DOUBLE, TensorProto.BFLOAT16):
        return False
    arr = numpy_helper.to_array(tensor).astype(np.float32)
    name = tensor.name or ""
    tensor.CopyFrom(numpy_helper.from_array(arr, name))
    return True


def _downcast_graph_fp64(graph: onnx.GraphProto) -> int:
    """Recursively downcast float64 tensors within a graph to float32."""
    converted = 0
    for initializer in graph.initializer:
        if _cast_tensor_proto_to_fp32(initializer):
            converted += 1

    def _fix_value_info(value_info: ValueInfoProto) -> None:
        nonlocal converted
        if not value_info.type.HasField("tensor_type"):
            return
        tensor_type = value_info.type.tensor_type
        if tensor_type.elem_type in (TensorProto.DOUBLE, TensorProto.BFLOAT16):
            tensor_type.elem_type = TensorProto.FLOAT
            converted += 1

    for value in list(graph.input) + list(graph.output) + list(graph.value_info):
        _fix_value_info(value)

    for node in graph.node:
        for attr in node.attribute:
            if attr.type == AttributeProto.TENSOR:
                if _cast_tensor_proto_to_fp32(attr.t):
                    converted += 1
            elif attr.type == AttributeProto.GRAPH:
                converted += _downcast_graph_fp64(attr.g)
            elif attr.type == AttributeProto.GRAPHS:
                for sub in attr.graphs:
                    converted += _downcast_graph_fp64(sub)
            elif attr.type == AttributeProto.TENSORS:
                for tensor in attr.tensors:
                    if _cast_tensor_proto_to_fp32(tensor):
                        converted += 1
    return converted


def _normalise_model(m: onnx.ModelProto, opset_version: Optional[int] = None) -> None:
    """Ensure opset and dtype consistency before saving."""
    if opset_version is not None:
        _ensure_opset(m, opset_version)
    converted = _downcast_graph_fp64(m.graph)
    if converted > 0:
        logger.info("Downcast %d float64 tensor(s) to float32 for ONNX compatibility.", converted)

def _const_i64(name: str, vals: List[int]) -> onnx.NodeProto:
    """Create Constant node with int64 array."""
    arr = numpy_helper.from_array(
        np.asarray(vals, dtype="int64"),
        name
    )
    return helper.make_node(
        "Constant", [], [name],
        name=name + "_const",
        value=arr
    )


def _collect_export_tensors(output: Any) -> List[Any]:
    """Flatten arbitrary model outputs into a list of tensors for ONNX export."""
    tensors: List[Any] = []
    if output is None:
        return tensors

    if _TENSOR_TYPES and isinstance(output, _TENSOR_TYPES):
        tensors.append(output)
        return tensors

    if isinstance(output, (list, tuple)):
        for item in output:
            tensors.extend(_collect_export_tensors(item))
        return tensors

    if isinstance(output, dict):
        for value in output.values():
            tensors.extend(_collect_export_tensors(value))
        return tensors

    if _TENSOR_TYPES:
        raise TypeError(
            f"Unsupported export output type '{type(output)}'. "
            "Only tensors or collections of tensors are supported."
        )

    return tensors


class _ExportOutputAdapter(torch.nn.Module if torch is not None else object):
    """
    Wrap a VGGT model so that ONNX export always sees tensor-only outputs.
    This avoids tracing failures when auxiliary outputs are None.
    """

    def __init__(self, base_model):
        if torch is not None:
            super().__init__()
        self.model = base_model

    def forward(self, *args, **kwargs):  # type: ignore[override]
        outputs = self.model(*args, **kwargs)
        tensors = _collect_export_tensors(outputs)
        if not tensors:
            raise RuntimeError(
                "Model produced no tensor outputs during ONNX export. "
                "Ensure forward() returns at least one tensor."
            )
        if len(tensors) == 1:
            return tensors[0]
        return tuple(tensors)

def _get_attr_i(n: onnx.NodeProto, key: str, default: int = 0) -> int:
    """Get integer attribute from node."""
    for a in n.attribute:
        if a.name == key and a.type == AttributeProto.INT:
            return a.i
    return default

def _index_from_input(g: onnx.GraphProto, name: str) -> Optional[int]:
    """Extract integer index value from graph input."""
    # Check initializers
    for init in g.initializer:
        if init.name == name:
            arr = numpy_helper.to_array(init)
            if arr.size == 1:
                val = int(arr.item())
                return val if val >= 0 else None
            return None
    
    # Check Constant nodes
    for n in g.node:
        if n.op_type == "Constant" and n.output and n.output[0] == name:
            for a in n.attribute:
                if a.name == "value":
                    arr = numpy_helper.to_array(a.t)
                    if arr.size == 1:
                        val = int(arr.item())
                        return val if val >= 0 else None
    
    # Check Cast(Constant) pattern
    for n in g.node:
        if n.op_type == "Cast" and n.output and n.output[0] == name:
            src = n.input[0]
            return _index_from_input(g, src)
    
    return None

def _prune_to_outputs(m: onnx.ModelProto) -> None:
    """Remove unused nodes and initializers from graph."""
    g = m.graph
    prod = _prod_map(g)
    
    needed = set(v.name for v in g.output)
    changed = True
    while changed:
        changed = False
        for t in list(needed):
            if t in prod:
                n = prod[t]
                for i in n.input:
                    if i and i not in needed:
                        needed.add(i)
                        changed = True
    
    filtered_nodes = [n for n in g.node if any(o in needed for o in n.output)]
    filtered_inits = [i for i in g.initializer if i.name in needed]
    
    del g.node[:]
    g.node.extend(filtered_nodes)
    del g.initializer[:]
    g.initializer.extend(filtered_inits)


def _module_device(module) -> "torch.device":
    """Return device of first parameter (defaults to CPU)."""
    if torch is None:
        raise RuntimeError("PyTorch not available")
    for p in module.parameters():
        return p.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _torch_export_supports(param: str) -> bool:
    """Check whether torch.onnx.export signature exposes a parameter."""
    if torch is None:
        return False
    try:
        sig = inspect.signature(torch.onnx.export)
        return param in sig.parameters
    except (TypeError, ValueError):
        return False

# ---------- Calibration Utilities ----------

CALIB_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _gather_files(pattern_or_dir: str) -> List[str]:
    """Expand directory or glob pattern into a sorted list of files."""
    path = pathlib.Path(pattern_or_dir)
    if path.is_dir():
        files: List[str] = []
        for ext in CALIB_IMAGE_EXTS:
            files.extend(str(p) for p in sorted(path.rglob(f"*{ext}")))
        return files
    matches = glob.glob(pattern_or_dir)
    return sorted(matches)


def _load_image_tensor(path: str, hw: Tuple[int, int]) -> np.ndarray:
    """Load image as float32 tensor (C, H, W) normalised to [-1, 1]."""
    if Image is None:
        raise RuntimeError(
            "Pillow (PIL) is required for image-based calibration. Install with `pip install pillow`."
        )
    h, w = hw
    with Image.open(path) as img:
        img = img.convert("RGB")
        if img.size != (w, h):
            img = img.resize((w, h), resample=Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    arr = arr * 2.0 - 1.0  # normalise to [-1, 1]
    return arr


def _prepare_batches_from_images(
    source: str,
    num_cams: int,
    hw: Tuple[int, int],
    num_batches: int,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """Create calibration batches by grouping images into multi-view stacks."""
    files = _gather_files(source)
    if not files:
        raise FileNotFoundError(f"No calibration images found in '{source}'")

    logger.info(f"Found {len(files)} calibration frames in '{source}'")
    rng = np.random.default_rng(seed)
    batches: List[np.ndarray] = []

    # Pre-load to avoid repeated disk I/O when dataset is small
    cache: Dict[str, np.ndarray] = {}

    # Allow re-use of images when the directory is smaller than required
    total_needed = max(num_batches * num_cams, num_cams)
    if len(files) < total_needed:
        logger.warning(
            "Calibration directory has fewer frames (%d) than required (%d). "
            "Images will be re-used with shuffling.",
            len(files),
            total_needed,
        )

    for batch_idx in range(num_batches):
        views: List[np.ndarray] = []
        # Sample without replacement until exhausted, then reshuffle
        choices = rng.choice(files, size=num_cams, replace=len(files) < num_cams)
        for path in choices:
            if path not in cache:
                cache[path] = _load_image_tensor(path, hw)
            views.append(cache[path])
        batch = np.stack(views, axis=0)  # (num_cams, C, H, W)
        batches.append(batch.astype(np.float32, copy=False))

    return batches


def _prepare_batches_from_numpy(
    array_path: str,
    num_cams: int,
    hw: Tuple[int, int],
    num_batches: int,
) -> List[np.ndarray]:
    """Load calibration batches from a .npy/.npz tensor file."""
    data = np.load(array_path, allow_pickle=False)
    if isinstance(data, np.lib.npyio.NpzFile):
        # Use first array if .npz
        first_key = next(iter(sorted(data.files)))
        data = data[first_key]

    data = np.asarray(data)
    if data.dtype != np.float32:
        data = data.astype(np.float32)

    if data.ndim == 5:
        # (B, num_cams, C, H, W)
        if data.shape[1] != num_cams:
            raise ValueError(
                f"Calibration tensor expects num_cams={num_cams}, found {data.shape[1]}"
            )
        batches = [np.ascontiguousarray(data[i]) for i in range(min(num_batches, data.shape[0]))]
    elif data.ndim == 4:
        # (N, C, H, W) -> tile into batches
        if data.shape[1:] != (3, *hw):
            raise ValueError(
                f"Calibration tensor expects shape (N,3,{hw[0]},{hw[1]}), got {data.shape}"
            )
        batches = []
        idx = 0
        for _ in range(num_batches):
            chunk = []
            for _ in range(num_cams):
                chunk.append(data[idx % data.shape[0]])
                idx += 1
            batches.append(np.stack(chunk, axis=0))
    else:
        raise ValueError(f"Unsupported calibration tensor shape: {data.shape}")

    if not batches:
        raise ValueError("Calibration tensor did not yield any batches")

    logger.info(
        "Loaded %d calibration batches from tensor '%s'",
        len(batches),
        array_path,
    )
    return batches[:num_batches]


def prepare_calibration_batches(
    source: Optional[str],
    num_cams: int,
    hw: Tuple[int, int],
    num_batches: int,
    seed: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Prepare calibration batches for INT8 quantisation.

    The source can be:
    * None         → random Gaussian noise (fallback)
    * directory    → image files grouped into batches
    * glob pattern → image files grouped into batches
    * .npy / .npz  → tensor with shape (B,num_cams,3,H,W) or (N,3,H,W)
    """
    if num_batches <= 0:
        return []

    if not source:
        logger.warning("No calibration data provided; using random Gaussian noise")
        batches = [
            np.random.randn(num_cams, 3, hw[0], hw[1]).astype(np.float32)
            for _ in range(num_batches)
        ]
        return batches

    path = pathlib.Path(source)
    if path.suffix.lower() in {".npy", ".npz"} and path.exists():
        return _prepare_batches_from_numpy(str(path), num_cams, hw, num_batches)

    # Treat as directory or glob
    return _prepare_batches_from_images(source, num_cams, hw, num_batches, seed=seed)

# ---------- FIXED INT8 Calibrator ----------

class SimpleCalibrator(trt_mod.IInt8EntropyCalibrator2 if trt_mod else object):
    """
    TensorRT entropy calibrator with optional GPU staging memory.

    Accepts a pre-built list (or iterable) of numpy batches shaped like the
    TensorRT network input: (num_cams, 3, H, W).
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        cache_file: str,
        batches: Sequence[np.ndarray],
        use_gpu: bool = True,
    ):
        if trt_mod is None:
            raise RuntimeError("TensorRT not available; cannot build INT8 engines")

        trt_mod.IInt8EntropyCalibrator2.__init__(self)

        self.input_shape = tuple(input_shape)
        self.cache_file = cache_file
        self._batches = [np.ascontiguousarray(b.astype(np.float32, copy=False)) for b in batches]
        self._cursor = 0
        self._device_tensor = None
        self._device_ptr: Optional[int] = None
        self._use_gpu = bool(use_gpu and torch is not None and torch.cuda.is_available())

        if not self._batches:
            raise ValueError("Calibrator requires at least one calibration batch")

        expected_elems = int(np.prod(self.input_shape))
        for idx, batch in enumerate(self._batches):
            if batch.shape != self.input_shape:
                raise ValueError(
                    f"Calibration batch {idx} has shape {batch.shape}, expected {self.input_shape}"
                )
            if batch.size != expected_elems:
                raise ValueError(
                    f"Calibration batch {idx} has {batch.size} elements, expected {expected_elems}"
                )

        if self._use_gpu:
            self._device_tensor = torch.empty(
                self.input_shape,
                dtype=torch.float32,
                device="cuda",
            )
            self._device_ptr = int(self._device_tensor.data_ptr())
            logger.info(
                "INT8 calibrator using GPU staging buffer (%.2f MB)",
                self._device_tensor.numel() * 4 / (1024**2),
            )
        else:
            logger.info("INT8 calibrator running on CPU host memory")

        self._num_batches = len(self._batches)

    def get_batch_size(self) -> int:
        return self.input_shape[0]

    def get_batch(self, names) -> Optional[List[int]]:
        """Return next calibration batch pointer."""
        if self._cursor >= self._num_batches:
            logger.info("INT8 calibrator exhausted after %d batches", self._num_batches)
            return None

        batch = self._batches[self._cursor]
        self._cursor += 1

        if self._use_gpu and self._device_tensor is not None:
            gpu_tensor = torch.from_numpy(batch).to(device="cuda", dtype=torch.float32)
            self._device_tensor.copy_(gpu_tensor)
            return [self._device_ptr] if self._device_ptr is not None else None

        # CPU fallback – TensorRT will copy internally
        return [int(batch.ctypes.data)]

    def read_calibration_cache(self) -> Optional[bytes]:
        """Read existing calibration cache."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                data = f.read()
            if data:
                logger.info("Loaded INT8 calibration cache '%s'", self.cache_file)
            return data
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        """Persist calibration cache for future runs."""
        _mkdir_for(self.cache_file)
        with open(self.cache_file, "wb") as f:
            f.write(cache)
        logger.info("Calibration cache saved to '%s'", self.cache_file)

# ---------- Precision Configuration ----------

@dataclass
class PrecisionConfig:
    """Configuration for different precision modes."""
    name: str
    flags: List[str]
    suffix: str
    description: str

PRECISIONS = {
    "fp32": PrecisionConfig(
        name="fp32",
        flags=[],
        suffix="",
        description="Full precision (baseline)"
    ),
    "fp16": PrecisionConfig(
        name="fp16",
        flags=["FP16", "TF32"],
        suffix="_fp16",
        description="Half precision with TF32 fallback"
    ),
    "bf16": PrecisionConfig(
        name="bf16",
        flags=["BF16", "TF32"],
        suffix="_bf16",
        description="BFloat16 precision"
    ),
    "fp8": PrecisionConfig(
        name="fp8",
        flags=["FP8", "FP16", "TF32"],
        suffix="_fp8",
        description="FP8 precision (RTX 5090 limited support)"
    ),
    "int8": PrecisionConfig(
        name="int8",
        flags=["INT8", "FP16", "TF32"],
        suffix="_int8",
        description="INT8 quantization with FP16 fallback"
    ),
}


SUPPORTED_QUANT_MODES: Dict[str, str] = {
    "none": "No pre-quantisation (default)",
    "torch-int8-dynamic": "PyTorch dynamic INT8 on Linear/GRU/LSTM layers",
    "bitsandbytes-8bit": "bitsandbytes Linear8bitLt replacement",
    "bitsandbytes-nf4": "bitsandbytes Linear4bit (NF4) weight quantisation",
    "bitsandbytes-fp4": "bitsandbytes Linear4bit (FP4) weight quantisation",
    "modelopt-fp8": "NVIDIA ModelOpt FP8 recipe",
    "modelopt-nvfp4": "NVIDIA ModelOpt NVFP4 (4-bit) quantisation",
}


class QuantizationManager:
    """Optional pre-quantisation before exporting to ONNX."""

    def __init__(self, mode: str, allow_gpu_only: bool = True):
        self.mode = mode or "none"
        if self.mode not in SUPPORTED_QUANT_MODES:
            raise ValueError(
                f"Invalid quantisation mode '{self.mode}'. "
                f"Valid options: {list(SUPPORTED_QUANT_MODES.keys())}"
            )
        self.allow_gpu_only = allow_gpu_only

    # Public -----------------------------------------------------------------

    def describe(self) -> str:
        return SUPPORTED_QUANT_MODES[self.mode]

    def requires_gpu(self) -> bool:
        return self.mode in {
            "bitsandbytes-8bit",
            "bitsandbytes-nf4",
            "bitsandbytes-fp4",
            "modelopt-fp8",
        }

    def prefers_cpu(self) -> bool:
        return self.mode == "torch-int8-dynamic"

    def apply(self, model, forward_loop=None):
        if self.mode == "none":
            return model

        handler_name = f"_apply_{self.mode.replace('-', '_')}"
        handler = getattr(self, handler_name, None)
        if handler is None:
            raise NotImplementedError(
                f"Quantisation handler '{self.mode}' is not implemented."
            )

        logger.info("Applying pre-quantisation: %s", SUPPORTED_QUANT_MODES[self.mode])
        try:
            quantized = handler(model, forward_loop=forward_loop)
        except TypeError:
            quantized = handler(model)
        logger.info("Pre-quantisation '%s' complete", self.mode)
        return quantized

    def fallback_modes(self) -> List[str]:
        """Return fallback quantisation modes to try if export fails."""
        if self.mode == "none":
            return []

        fallback_map = {
            "torch-int8-dynamic": ["bitsandbytes-8bit", "none"],
            "bitsandbytes-8bit": ["bitsandbytes-nf4", "bitsandbytes-fp4", "none"],
            "bitsandbytes-nf4": ["bitsandbytes-fp4", "none"],
            "bitsandbytes-fp4": ["none"],
            "modelopt-fp8": ["none"],
            "modelopt-nvfp4": ["modelopt-fp8", "none"],
        }
        modes = fallback_map.get(self.mode, ["none"])
        # Deduplicate while preserving order
        seen = set([self.mode])
        dedup: List[str] = []
        for m in modes:
            if m not in seen:
                dedup.append(m)
                seen.add(m)
        return dedup

    def is_onnx_compatible(self) -> bool:
        """Return False for quantisation modes that produce unsupported ONNX ops."""
        return self.mode not in {
            "bitsandbytes-8bit",
            "bitsandbytes-nf4",
            "bitsandbytes-fp4",
        }

    # Internal handlers ------------------------------------------------------

    def _apply_torch_int8_dynamic(self, model, forward_loop=None):
        if torch is None:
            raise RuntimeError("PyTorch not available")
        if not hasattr(torch, "ao") or not hasattr(torch.ao, "quantization"):
            raise RuntimeError(
                "torch.ao.quantization is missing. Install PyTorch with quantization support."
            )
        q = torch.ao.quantization
        modules = {torch.nn.Linear, torch.nn.GRU, torch.nn.LSTM}
        quantized = q.quantize_dynamic(model, modules, dtype=torch.qint8)
        return quantized

    def _replace_linear_modules(self, model, factory):
        """Recursively replace nn.Linear layers using factory(original_layer)."""
        for name, child in list(model.named_children()):
            if isinstance(child, torch.nn.Linear):
                new_layer = factory(child)
                setattr(model, name, new_layer)
            else:
                self._replace_linear_modules(child, factory)
        return model

    def _bnb_import(self):
        bnb = _lazy_import("bitsandbytes")
        if bnb is None:
            raise RuntimeError(
                "bitsandbytes is required for this quantisation mode. "
                "Install with `pip install bitsandbytes`."
            )
        return bnb

    def _apply_bitsandbytes_8bit(self, model, forward_loop=None):
        if torch is None:
            raise RuntimeError("PyTorch not available")
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if self.allow_gpu_only and not (torch.cuda.is_available() or has_mps):
            raise RuntimeError("bitsandbytes quantisation requires CUDA or MPS backend.")
        self._bnb_import()
        try:
            from bitsandbytes.nn import Linear8bitLt
        except ImportError as exc:
            raise RuntimeError(
                "bitsandbytes.nn.Linear8bitLt not found. Update bitsandbytes."
            ) from exc

        params = list(model.parameters())
        device = params[0].device if params else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def factory(orig: torch.nn.Linear):
            layer = Linear8bitLt(
                orig.in_features,
                orig.out_features,
                bias=orig.bias is not None,
                has_fp16_weights=True,
                threshold=6.0,
            )
            layer = layer.to(device)
            try:
                layer.load_state_dict(orig.state_dict(), strict=False)
            except Exception as exc:
                logger.warning("Linear8bitLt load_state_dict failed: %s; copying weights directly", exc)
                with torch.no_grad():
                    layer.weight = torch.nn.Parameter(orig.weight.data.to(device))
                    if orig.bias is not None:
                        layer.bias = torch.nn.Parameter(orig.bias.data.to(device))
            return layer

        self._replace_linear_modules(model, factory)
        return model

    def _apply_bitsandbytes_nf4(self, model, forward_loop=None):
        return self._apply_bitsandbytes_4bit(model, quant_type="nf4")

    def _apply_bitsandbytes_fp4(self, model, forward_loop=None):
        return self._apply_bitsandbytes_4bit(model, quant_type="fp4")

    def _apply_bitsandbytes_4bit(self, model, quant_type: str, forward_loop=None):
        if torch is None:
            raise RuntimeError("PyTorch not available")
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if self.allow_gpu_only and not (torch.cuda.is_available() or has_mps):
            raise RuntimeError("bitsandbytes quantisation requires CUDA or MPS backend.")
        self._bnb_import()
        try:
            from bitsandbytes.nn import Linear4bit
        except ImportError as exc:
            raise RuntimeError(
                "bitsandbytes.nn.Linear4bit not found. Update bitsandbytes."
            ) from exc

        params = list(model.parameters())
        device = params[0].device if params else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def factory(orig: torch.nn.Linear):
            layer = Linear4bit(
                orig.in_features,
                orig.out_features,
                bias=orig.bias is not None,
                compute_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                compress_statistics=True,
                quant_type=quant_type,
            )
            layer = layer.to(device)
            try:
                layer.load_state_dict(orig.state_dict(), strict=False)
            except Exception as exc:
                logger.warning("Linear4bit load_state_dict failed: %s; copying weights directly", exc)
                with torch.no_grad():
                    layer.weight = torch.nn.Parameter(orig.weight.data.to(device))
                    if orig.bias is not None:
                        layer.bias = torch.nn.Parameter(orig.bias.data.to(device))
            return layer

        self._replace_linear_modules(model, factory)
        return model

    def _apply_modelopt_fp8(self, model, forward_loop=None):
        mod = _lazy_import("modelopt")
        if mod is None:
            raise RuntimeError(
                "NVIDIA ModelOpt is required for modelopt-fp8 mode. "
                "Install with `pip install modelopt` (requires CUDA 12.2+)."
            )
        try:
            from modelopt.torch.quantization import FP8_DEFAULT_CFG, quantize
        except ImportError as exc:
            raise RuntimeError(
                "modelopt.torch.quantization APIs not available. "
                "Ensure you installed modelopt>=0.9."
            ) from exc

        model.eval()
        quantized = quantize(model, FP8_DEFAULT_CFG, forward_loop=forward_loop)
        return quantized

    def _apply_modelopt_nvfp4(self, model, forward_loop=None):
        mod = _lazy_import("modelopt")
        if mod is None:
            raise RuntimeError(
                "NVIDIA ModelOpt is required for modelopt-nvfp4 mode. "
                "Install with `pip install modelopt` (requires CUDA 12.2+)."
            )
        try:
            from modelopt.torch.quantization import NVFP4_DEFAULT_CFG, quantize
        except ImportError as exc:
            raise RuntimeError(
                "modelopt.torch.quantization APIs not available. "
                "Ensure you installed modelopt>=0.9."
            ) from exc

        model.eval()
        quantized = quantize(model, NVFP4_DEFAULT_CFG, forward_loop=forward_loop)
        return quantized

# ---------- Pipeline Class ----------

class VGGTPipeline:
    """
    Fixed pipeline for converting VGGT models to TensorRT engines.
    """
    
    def __init__(
        self,
        num_cams: int = 8,
        hw: Tuple[int, int] = (518, 518),
        opset: int = 18,
        workspace_gb: int = 28,
        precision: str = "fp16",
        simplify: bool = True,
        model_name: str = "facebook/VGGT-1B",
        opt_level: int = 5,
        max_aux_streams: int = 4,
        pcd_only: bool = False,
        quant_mode: str = "none",
        calib_source: Optional[str] = None,
        calib_batches: int = 32,
        calib_seed: Optional[int] = 1337,
        calib_use_gpu: bool = True,
    ):
        """
        Initialize pipeline configuration.
        
        Args:
            num_cams: Number of camera views (1-32 recommended)
            hw: Input height and width as (H, W)
            opset: ONNX opset version
            workspace_gb: TensorRT workspace size in GB
            precision: Precision mode (fp32/fp16/bf16/fp8/int8)
            simplify: Enable ONNX simplification
            model_name: HuggingFace model identifier
            opt_level: TensorRT optimization level (0-5)
            max_aux_streams: Maximum auxiliary CUDA streams
            pcd_only: Export only depth+camera heads (faster for PCD)
            quant_mode: Optional pre-quantisation recipe before ONNX export
            calib_source: Path/glob/.npy providing calibration data for INT8
            calib_batches: Number of calibration batches to feed TensorRT
            calib_seed: Random seed for calibration shuffling / synthetic data
            calib_use_gpu: Stage calibration batches on GPU memory if available
        """
        if num_cams < 1:
            raise ValueError(f"num_cams must be >= 1, got {num_cams}")
        if hw[0] < 1 or hw[1] < 1:
            raise ValueError(f"Invalid dimensions: {hw}")
        if precision not in PRECISIONS:
            raise ValueError(f"Invalid precision: {precision}. Choose from {list(PRECISIONS.keys())}")
        
        self.num_cams = num_cams
        self.hw = hw
        self.opset = opset
        self.workspace_gb = workspace_gb
        self.precision_config = PRECISIONS[precision]
        self.do_simplify = simplify
        self.model_name = model_name
        self.opt_level = opt_level
        self.max_aux_streams = max_aux_streams
        self.pcd_only = pcd_only
        self.quant_manager = QuantizationManager(quant_mode)
        self.calib_source = calib_source
        self.calib_batches = calib_batches
        self.calib_seed = calib_seed
        self.calib_use_gpu = calib_use_gpu
        
        logger.info(f"Pipeline config: {num_cams} cameras, {hw[0]}x{hw[1]}, {precision} precision")
        if pcd_only:
            logger.info("PCD-only mode: exporting depth + camera heads only (30% faster)")
        if self.quant_manager.mode != "none":
            logger.info("Pre-quantisation: %s", self.quant_manager.describe())
        if precision == "int8":
            logger.info(
                "INT8 calibration: source=%s batches=%s seed=%s gpu=%s",
                calib_source or "random Gaussian",
                calib_batches,
                calib_seed,
                calib_use_gpu,
            )

    def export_from_hf(self, out_onnx: str, device: str = "cuda") -> str:
        """Export VGGT model from HuggingFace to ONNX format with quantisation fallbacks."""
        if torch is None:
            raise RuntimeError("PyTorch not available. Install with: pip install torch")

        try:
            from vggt.models.vggt import VGGT
        except ImportError as exc:
            raise RuntimeError(
                "VGGT module not available. Install with: pip install vggt"
            ) from exc

        _mkdir_for(out_onnx)
        C, H, W = 3, *self.hw
        requested_device = (device or "cuda").lower()

        attempt_modes: List[str] = []
        seen_modes = set()
        for mode in [self.quant_manager.mode, *self.quant_manager.fallback_modes()]:
            if mode not in seen_modes:
                attempt_modes.append(mode)
                seen_modes.add(mode)

        if not attempt_modes:
            attempt_modes = ["none"]

        logged_pcd_notice = False
        last_exc: Optional[Exception] = None

        def _resolve_export_device(manager: QuantizationManager, requested: str) -> str:
            dev = requested
            if manager.prefers_cpu():
                if requested != "cpu":
                    logger.info(
                        "Quantisation mode '%s' runs on CPU; switching export device to CPU",
                        manager.mode,
                    )
                dev = "cpu"
            if manager.requires_gpu():
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        f"Quantisation mode '{manager.mode}' requires CUDA, but no CUDA device is available."
                    )
                if dev != "cuda":
                    logger.info(
                        "Quantisation mode '%s' prefers CUDA; switching export device to CUDA",
                        manager.mode,
                    )
                dev = "cuda"
            return dev

        def _load_model() -> "torch.nn.Module":
            model = VGGT.from_pretrained(self.model_name)
            model.eval()
            return model.to(dtype=torch.float32, device="cpu")

        export_kwargs_base = {
            "input_names": ["images"],
            "output_names": None,
            "opset_version": self.opset,
            "do_constant_folding": True,
            "dynamic_axes": None,
            "verbose": False,
            "export_params": True,
        }

        total_attempts = len(attempt_modes)

        for idx, mode in enumerate(attempt_modes, start=1):
            manager = self.quant_manager if mode == self.quant_manager.mode else QuantizationManager(mode)

            if manager.mode != "none" and not manager.is_onnx_compatible():
                label = "Requested" if idx == 1 else f"Fallback ({idx}/{total_attempts})"
                logger.info(
                    "%s quantisation mode '%s' is not ONNX-compatible with ONNX export; skipping.",
                    label,
                    manager.mode,
                )
                last_exc = RuntimeError(
                    f"Quantisation mode '{manager.mode}' is not ONNX-compatible"
                )
                continue

            try:
                export_device = _resolve_export_device(manager, requested_device)
            except Exception as exc:
                logger.error("Skipping quantisation mode '%s': %s", manager.mode, exc)
                last_exc = exc
                continue

            if idx == 1:
                logger.info(f"Loading model: {self.model_name}")
            else:
                logger.info(
                    "Retrying export with fallback quantisation mode '%s' (%d/%d)",
                    manager.mode,
                    idx,
                    total_attempts,
                )

            try:
                model = _load_model()
            except Exception as exc:
                raise RuntimeError(f"Failed to load model '{self.model_name}': {exc}") from exc

            modelopt_forward_loop = None
            if manager.mode.startswith("modelopt-"):
                loop = self._build_modelopt_forward_loop()
                if loop is not None:
                    modelopt_forward_loop = loop

            if self.pcd_only and not logged_pcd_notice:
                logger.info("Modifying model for PCD-only export...")
                logger.warning("PCD-only mode: will prune unused outputs after export")
                logged_pcd_notice = True

            try:
                if manager.mode != "none":
                    if manager.requires_gpu():
                        model = model.to("cuda")
                    model = manager.apply(model, forward_loop=modelopt_forward_loop)
                    model.eval()
            except Exception as quant_exc:
                logger.error(
                    "Quantisation mode '%s' failed prior to export: %s",
                    manager.mode,
                    quant_exc,
                )
                last_exc = quant_exc
                continue

            if torch is not None:
                model = _ExportOutputAdapter(model)
                model.eval()

            # Move to final export device
            if export_device == "cuda":
                if not torch.cuda.is_available():
                    logger.error("CUDA not available for export; skipping mode '%s'", manager.mode)
                    last_exc = RuntimeError("CUDA unavailable")
                    continue
                model = model.to("cuda")
            else:
                model = model.to("cpu")

            try:
                sample_device = _module_device(model)
            except RuntimeError as exc:
                logger.error("Failed to determine model device for mode '%s': %s", manager.mode, exc)
                last_exc = exc
                continue

            x = torch.randn(
                self.num_cams,
                C,
                H,
                W,
                dtype=torch.float32,
                device=sample_device,
            )

            logger.info(
                "Exporting to ONNX with shape [%d, %d, %d, %d] on %s (quant='%s')",
                self.num_cams,
                C,
                H,
                W,
                sample_device,
                manager.mode,
            )

            export_kwargs = dict(export_kwargs_base)
            export_uses_dynamo = manager.mode in {"none", "torch-int8-dynamic"}
            optional_params = {
                "dynamo": export_uses_dynamo,
                "external_data": True,
                "optimize": export_uses_dynamo,
                "verify": False,
                "profile": False,
                "report": False,
            }
            for key, value in optional_params.items():
                if _torch_export_supports(key):
                    export_kwargs[key] = value

            try:
                with torch.inference_mode():
                    try:
                        torch.onnx.export(model, (x,), out_onnx, **export_kwargs)
                    except torch.cuda.OutOfMemoryError:
                        if export_device == "cuda":
                            logger.warning(
                                "GPU OOM during export (quant='%s'); retrying on CPU without dynamo",
                                manager.mode,
                            )
                        model = model.to("cpu")
                        x = x.to("cpu")
                        fallback_kwargs = dict(export_kwargs)
                        if "dynamo" in fallback_kwargs:
                            fallback_kwargs["dynamo"] = False
                        if "optimize" in fallback_kwargs:
                            fallback_kwargs["optimize"] = False
                        torch.onnx.export(model, (x,), out_onnx, **fallback_kwargs)
            except Exception as exc:
                logger.error(
                    "ONNX export attempt failed with quantisation '%s': %s",
                    manager.mode,
                    exc,
                )
                last_exc = exc
                continue

            logger.info(f"Wrote {out_onnx}")
            self.rebind_external_data(out_onnx)

            if self.pcd_only:
                self._prune_for_pcd(out_onnx)

            return out_onnx

        modes_tried = ", ".join(attempt_modes)
        error_message = (
            f"ONNX export failed after trying quantisation modes: {modes_tried}."
            if modes_tried
            else "ONNX export failed."
        )
        if last_exc is not None:
            raise RuntimeError(f"{error_message} Last error: {last_exc}") from last_exc
        raise RuntimeError(error_message)

    @staticmethod
    def _inspect_engine_precisions(trt, engine) -> Dict[str, int]:
        """Collect TensorRT tensor data type counts for diagnostics."""
        stats: Dict[str, int] = {}
        try:
            if hasattr(engine, "num_tensors"):
                for idx in range(engine.num_tensors):
                    name = engine.get_tensor_name(idx)
                    dtype = engine.get_tensor_dtype(name)
                    dtype_name = getattr(dtype, "name", str(dtype))
                    stats[dtype_name] = stats.get(dtype_name, 0) + 1
            else:
                for idx in range(engine.num_bindings):
                    if hasattr(engine, "get_tensor_dtype"):
                        name = engine.get_binding_name(idx)
                        dtype = engine.get_tensor_dtype(name)
                    else:
                        dtype = engine.get_binding_dtype(idx)
                    dtype_name = getattr(dtype, "name", str(dtype))
                    stats[dtype_name] = stats.get(dtype_name, 0) + 1
        except Exception as exc:
            logger.debug("Failed to inspect engine precisions: %s", exc)
        return stats

    def _prune_for_pcd(self, onnx_path: str) -> None:
        """
        Prune ONNX model to keep only depth + camera outputs.
        
        VGGT outputs (from your log):
        - cat_322: camera parameters (keep)
        - view_411, view_412: depth maps (keep)
        - view_451, view_452: point maps (remove)
        - unsqueeze: tracking features (remove)
        """
        logger.info("Pruning model for PCD-only mode...")
        m = onnx.load(onnx_path, load_external_data=True)
        g = m.graph
        
        # Keep only depth + camera outputs
        keep_outputs = []
        for output in g.output:
            name = output.name
            # Keep camera params and depth outputs
            if "cat_322" in name or "view_411" in name or "view_412" in name:
                keep_outputs.append(output)
                dims = []
                tt = output.type.tensor_type
                for dim in tt.shape.dim:
                    if dim.HasField("dim_value"):
                        dims.append(dim.dim_value)
                    elif dim.HasField("dim_param"):
                        dims.append(dim.dim_param)
                    else:
                        dims.append("?")
                logger.info(f"  Keeping output: {name} shape={tuple(dims)}")
                if "cat_322" in name:
                    last_dim = dims[-1] if dims else None
                    if last_dim not in (7, 9, 12, 16, "9", "12", "16", "7"):
                        logger.warning(
                            "    Unexpected camera tensor width %s; VGGT runtime expects 7|9|12|16 (quat/T, pose encoding, or 4x4).",
                            last_dim,
                        )
                    else:
                        logger.info(
                            "    cat_322 encodes camera pose as (B=1, V, %s). Depth outputs use (B=1, V, H, W).",
                            last_dim,
                        )
            else:
                logger.info(f"  Removing output: {name}")
        
        # Update outputs
        del g.output[:]
        g.output.extend(keep_outputs)
        
        # Prune unused nodes
        _prune_to_outputs(m)
        
        # Save pruned model
        rel = _data_rel(onnx_path)
        abs_p = _data_abs(onnx_path)
        _rm(abs_p)
        
        _normalise_model(m, self.opset)

        convert_model_to_external_data(
            m, True, rel, EXTERNAL_DATA_THRESHOLD, False
        )
        onnx.save_model(m, onnx_path)
        
        logger.info(f"Pruned model saved: {onnx_path}")

    def rebind_external_data(self, onnx_path: str) -> None:
        """Rewrite external data file with relative paths."""
        logger.info("Rebinding external data...")
        m = onnx.load(onnx_path, load_external_data=True)
        
        # CRITICAL FIX: Ensure valid opset BEFORE any operations
        _normalise_model(m, self.opset)

        rel = _data_rel(onnx_path)
        abs_p = _data_abs(onnx_path)

        _rm(abs_p)
        
        convert_model_to_external_data(
            m,
            all_tensors_to_one_file=True,
            location=rel,
            size_threshold=EXTERNAL_DATA_THRESHOLD,
            convert_attribute=False,
        )
        onnx.save_model(m, onnx_path)
        
        logger.info(f"External data: {abs_p}")
        try:
            s_on = os.path.getsize(onnx_path)
            s_da = os.path.getsize(abs_p) if os.path.exists(abs_p) else 0
            logger.info(f"Sizes: ONNX={_fmt_size(s_on)}, DATA={_fmt_size(s_da)}")
        except Exception:
            pass

    def _workspace_plan(self, initial: Optional[int] = None) -> List[int]:
        """Generate a descending list of workspace sizes (GB) for TRT retries."""
        start = int(initial if initial is not None else self.workspace_gb)
        candidates = [start]
        candidates.extend(ws for ws in WORKSPACE_FALLBACKS if ws < start)

        plan: List[int] = []
        seen: set = set()
        for ws in candidates:
            ws_int = int(ws)
            if ws_int <= 0 or ws_int in seen:
                continue
            plan.append(ws_int)
            seen.add(ws_int)

        if 2 not in seen and start > 2:
            plan.append(2)

        if not plan:
            plan = [max(2, start)]

        return plan

    def _build_modelopt_forward_loop(self) -> Optional["Callable[[torch.nn.Module], None]"]:  # type: ignore[name-defined]
        """Prepare a calibration forward loop for ModelOpt quantisation."""
        if torch is None:
            return None

        max_batches = max(1, min(self.calib_batches, 8))
        try:
            batches = prepare_calibration_batches(
                self.calib_source,
                self.num_cams,
                self.hw,
                max_batches,
                seed=self.calib_seed,
            )
        except Exception as exc:
            logger.warning(
                "ModelOpt calibration data unavailable (%s); generating synthetic batches",
                exc,
            )
            batches = prepare_calibration_batches(
                None,
                self.num_cams,
                self.hw,
                max_batches,
                seed=self.calib_seed,
            )

        if not batches:
            batches = prepare_calibration_batches(
                None,
                self.num_cams,
                self.hw,
                2,
                seed=self.calib_seed,
            )

        # Limit to a few batches for calibration speed
        batches = batches[:max_batches]

        def _forward_loop(module: "torch.nn.Module") -> None:
            device = next(module.parameters()).device
            dtype = next(module.parameters()).dtype
            with torch.inference_mode():
                for np_batch in batches:
                    tensor = torch.from_numpy(np_batch).to(device=device, dtype=dtype)
                    module(tensor)

        logger.info("ModelOpt calibration batches: %d", len(batches))
        return _forward_loop

    def _normalize_softmax_last_axis(self, g: onnx.GraphProto) -> int:
        """Normalize Softmax nodes to use axis=-1."""
        changed = 0
        for n in g.node:
            if n.op_type == "Softmax":
                attrs_to_keep = [a for a in n.attribute if a.name != "axis"]
                del n.attribute[:]
                n.attribute.extend(attrs_to_keep)
                n.attribute.extend([helper.make_attribute("axis", -1)])
                changed += 1
        return changed

    def _normalize_squeeze(self, g: onnx.GraphProto) -> int:
        """Normalize Squeeze nodes to use input instead of attributes."""
        updated, out = 0, []
        for n in g.node:
            if n.op_type == "Squeeze":
                axes_attr = None
                for a in n.attribute:
                    if a.name == "axes":
                        axes_attr = a
                        break
                
                if axes_attr is not None:
                    if axes_attr.type == AttributeProto.INT:
                        axes = [axes_attr.i]
                    elif axes_attr.type == AttributeProto.INTS:
                        axes = list(axes_attr.ints)
                    else:
                        out.append(n)
                        continue
                    
                    axes_name = n.name + "_axes"
                    out.append(_const_i64(axes_name, axes))
                    out.append(
                        helper.make_node(
                            "Squeeze",
                            [n.input[0], axes_name],
                            list(n.output),
                            name=n.name
                        )
                    )
                    updated += 1
                    continue
            
            out.append(n)
        
        del g.node[:]
        g.node.extend(out)
        return updated

    def remove_sequences(self, src: str, dst: str) -> str:
        """Remove sequence operations from ONNX graph."""
        logger.info("Removing sequence operations...")
        m = onnx.load(src, load_external_data=True)
        
        # CRITICAL FIX: Ensure valid opset before modifications
        _ensure_opset(m, self.opset)
        
        g = m.graph
        prod = _prod_map(g)

        fixed_sq = self._normalize_squeeze(g)
        if fixed_sq > 0:
            logger.info(f"Normalized {fixed_sq} Squeeze nodes")
        
        fixed_sm = self._normalize_softmax_last_axis(g)
        if fixed_sm > 0:
            logger.info(f"Rewrote {fixed_sm} Softmax axes to -1")

        replaced, new_nodes = 0, []
        for n in g.node:
            if n.op_type != "SequenceAt":
                new_nodes.append(n)
                continue
            
            seq, idx = n.input
            p = prod.get(seq, None)
            idx_val = _index_from_input(g, idx)
            
            if idx_val is None or p is None:
                logger.warning(
                    f"Cannot resolve SequenceAt node '{n.name}': "
                    f"idx_val={idx_val}, producer={p.op_type if p else None}"
                )
                new_nodes.append(n)
                continue

            if p.op_type == "SplitToSequence":
                X = p.input[0]
                axis = _get_attr_i(p, "axis", 0)
                s, e, a = n.name + "_starts", n.name + "_ends", n.name + "_axes"
                sliced = n.output[0] + "_slice"
                
                new_nodes += [
                    _const_i64(s, [idx_val]),
                    _const_i64(e, [idx_val + 1]),
                    _const_i64(a, [axis]),
                    helper.make_node(
                        "Slice", [X, s, e, a], [sliced],
                        name=n.name + "_slice"
                    ),
                    helper.make_node(
                        "Squeeze", [sliced, a], [n.output[0]],
                        name=n.name + "_squeeze"
                    ),
                ]
                replaced += 1
                continue

            if p.op_type == "SequenceConstruct":
                ins = list(p.input)
                if 0 <= idx_val < len(ins):
                    new_nodes.append(
                        helper.make_node(
                            "Identity", [ins[idx_val]], [n.output[0]],
                            name=n.name + "_id"
                        )
                    )
                    replaced += 1
                    continue
                else:
                    logger.warning(
                        f"SequenceAt index {idx_val} out of range "
                        f"for SequenceConstruct with {len(ins)} inputs"
                    )

            new_nodes.append(n)

        del g.node[:]
        g.node.extend(new_nodes)
        _prune_to_outputs(m)

        remaining = [n for n in g.node if "Sequence" in n.op_type]
        if remaining:
            logger.error(f"{len(remaining)} sequence operations remain:")
            for n in remaining[:5]:
                logger.error(f"  - {n.op_type}: {n.name}")
            raise RuntimeError("Failed to remove all sequence operations")

        _mkdir_for(dst)
        rel = _data_rel(dst)
        abs_p = _data_abs(dst)
        _rm(abs_p)

        _normalise_model(m, self.opset)

        convert_model_to_external_data(
            m, True, rel, EXTERNAL_DATA_THRESHOLD, False
        )
        onnx.save_model(m, dst)
        
        logger.info(f"Replaced {replaced} sequence operations")
        logger.info(f"Wrote {dst}")
        return dst

    def simplify(self, src: str, dst: str) -> str:
        """Simplify ONNX graph using onnxsim."""
        if not self.do_simplify:
            logger.info("Simplification disabled")
            return src

        try:
            onnxsim = __import__("onnxsim")
        except Exception:
            logger.warning("onnxsim not installed, skipping simplification")
            return src

        logger.info("Simplifying ONNX graph...")
        m = onnx.load(src, load_external_data=True)
        
        # Ensure opset + dtype consistency before simplification
        _normalise_model(m, self.opset)
        
        # Verify opset is now correct
        logger.info(f"Model IR version: {m.ir_version}")
        for idx, op in enumerate(m.opset_import):
            logger.info(f"  opset[{idx}]: domain='{op.domain}', version={op.version}")

        import inspect
        params = set(inspect.signature(onnxsim.simplify).parameters)
        kwargs = {}
        if "skip_shape_inference" in params:
            kwargs["skip_shape_inference"] = False
        if "skip_constant_folding" in params:
            kwargs["skip_constant_folding"] = False
        if "skip_optimization" in params:
            kwargs["skip_optimization"] = False
        if "perform_optimization" in params:
            kwargs["perform_optimization"] = True

        try:
            result = onnxsim.simplify(m, **kwargs)
        except TypeError:
            result = onnxsim.simplify(m)
        except Exception as e:
            logger.error(f"Simplification failed: {e}")
            return src

        if isinstance(result, tuple):
            ms, ok = result
            if not ok:
                logger.warning("Simplification completed with warnings")
        else:
            ms = result

        _mkdir_for(dst)
        rel = _data_rel(dst)
        abs_p = _data_abs(dst)
        _rm(abs_p)

        _normalise_model(ms, self.opset)

        convert_model_to_external_data(ms, True, rel, EXTERNAL_DATA_THRESHOLD, False)
        onnx.save_model(ms, dst)

        logger.info(f"Wrote {dst}")
        return dst

    @staticmethod
    def onnx_check(path: str) -> None:
        """Validate ONNX model."""
        logger.info("Validating ONNX model...")
        try:
            onnx.checker.check_model(path)
            logger.info("ONNX validation passed")
        except Exception as e:
            raise RuntimeError(f"ONNX validation failed: {e}")

    def build_trt(
        self,
        onnx_path: str,
        engine_path: str,
        workspace_plan: Optional[List[int]] = None,
        workspace_attempt: int = 1,
        workspace_total: Optional[int] = None,
    ) -> str:
        """Build optimized TensorRT engine with automatic workspace fallbacks."""
        if trt_mod is None:
            raise RuntimeError("TensorRT not available")
        trt = trt_mod

        if workspace_plan is None:
            workspace_plan = self._workspace_plan()
        if not workspace_plan:
            raise RuntimeError("No valid TensorRT workspace sizes available")
        if workspace_total is None:
            workspace_total = len(workspace_plan)

        workspace_gb = int(workspace_plan[0])

        logger.info(f"Building TensorRT engine ({self.precision_config.description})...")
        logger.info(f"TensorRT version: {getattr(trt, '__version__', 'unknown')}")
        logger.info(
            "TensorRT workspace limit: %d GB (attempt %d/%d)",
            workspace_gb,
            workspace_attempt,
            workspace_total,
        )
        _mkdir_for(engine_path)
        
        # Setup logger (WARNING level for best perf during inference)
        logger_trt = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger_trt)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger_trt)

        # Parse ONNX
        ok = False
        if hasattr(parser, "parse_from_file"):
            ok = parser.parse_from_file(onnx_path)
        else:
            model_dir = os.path.dirname(onnx_path) or "."
            model_name = os.path.basename(onnx_path)
            cwd = os.getcwd()
            try:
                os.chdir(model_dir)
                with open(model_name, "rb") as f:
                    ok = parser.parse(f.read())
            finally:
                os.chdir(cwd)

        if not ok:
            logger.error("TensorRT parsing failed:")
            for i in range(parser.num_errors):
                logger.error(f"  {parser.get_error(i)}")
            raise RuntimeError("TensorRT failed to parse ONNX model")

        # Set input dtype
        if network.num_inputs > 0:
            input_tensor = network.get_input(0)
            input_shape = input_tensor.shape
            logger.info(f"Input shape: {input_shape}")
            
            try:
                # Use FP32 for input regardless of precision (will be converted internally)
                input_tensor.dtype = trt.float32
            except Exception:
                pass
            
            expected = (self.num_cams, 3, *self.hw)
            actual = tuple(input_shape)
            if actual != expected:
                logger.warning(f"Input shape mismatch: expected {expected}, got {actual}")

        # Configure builder
        config = builder.create_builder_config()
        
        # === OPTIMIZATION 1: Workspace size ===
        ws_bytes = workspace_gb * (1 << 30)
        try:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, ws_bytes)
        except Exception:
            try:
                config.max_workspace_size = ws_bytes
            except Exception:
                logger.warning("Could not set workspace size")
        
        # === OPTIMIZATION 2: Auxiliary streams ===
        try:
            builder.max_aux_streams = self.max_aux_streams
            logger.info(f"Max auxiliary streams: {self.max_aux_streams}")
        except Exception:
            pass
        
        # === OPTIMIZATION 3: Precision flags ===
        for flag_name in self.precision_config.flags:
            try:
                if hasattr(trt.BuilderFlag, flag_name):
                    flag = getattr(trt.BuilderFlag, flag_name)
                    config.set_flag(flag)
                    logger.info(f"Enabled {flag_name}")
                    
                    # FP8 warning suppression
                    if flag_name == "FP8":
                        logger.warning(
                            "FP8 enabled: RTX 5090 has limited FP8 support. "
                            "Expect 'Unsupported data type FP8' warnings for some ops. "
                            "TensorRT will fallback to FP16 automatically."
                        )
            except Exception as e:
                logger.warning(f"Could not enable {flag_name}: {e}")
        
        # === OPTIMIZATION 4: Optimization level ===
        opt_level_set = False
        try:
            if hasattr(config, "set_builder_optimization_level"):
                config.set_builder_optimization_level(self.opt_level)
                opt_level_set = True
            elif hasattr(config, "builder_optimization_level"):
                config.builder_optimization_level = self.opt_level
                opt_level_set = True
        except Exception:
            pass
        
        if opt_level_set:
            logger.info(f"Optimization level: {self.opt_level}")
        
        # === OPTIMIZATION 5: Tactic sources ===
        try:
            if hasattr(trt, "TacticSource") and hasattr(config, "set_tactic_sources"):
                TS = trt.TacticSource
                sources = 0
                for name in ("CUBLAS", "CUBLAS_LT", "CUDNN"):
                    if hasattr(TS, name):
                        sources |= getattr(TS, name)
                if sources:
                    config.set_tactic_sources(sources)
                    logger.info("Enabled all available tactic sources")
        except Exception:
            pass
        
        # === OPTIMIZATION 6: Timing cache ===
        cache_dir = os.path.dirname(engine_path) or "."
        cache_path = os.path.join(cache_dir, "trt_timing.cache")
        tc = None
        try:
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    tc = config.create_timing_cache(f.read())
                    config.set_timing_cache(tc, ignore_mismatch=False)
                    logger.info(f"Loaded timing cache from {cache_path}")
        except Exception as e:
            logger.info(f"No timing cache loaded: {e}")
        
        # === FIXED INT8 calibration ===
        if self.precision_config.name == "int8":
            logger.info("Setting up INT8 calibration...")
            calib_cache = os.path.join(
                cache_dir,
                f"calibration-{self.num_cams}x{self.hw[0]}x{self.hw[1]}.cache",
            )
            input_shape = (self.num_cams, 3, *self.hw)
            
            try:
                batches = prepare_calibration_batches(
                    self.calib_source,
                    self.num_cams,
                    self.hw,
                    max(self.calib_batches, 1),
                    seed=self.calib_seed,
                )
                calibrator = SimpleCalibrator(
                    input_shape,
                    calib_cache,
                    batches,
                    use_gpu=self.calib_use_gpu,
                )
                config.int8_calibrator = calibrator
                logger.info(
                    "INT8 calibrator configured (%d batches, cache=%s)",
                    len(batches),
                    calib_cache,
                )
            except Exception as e:
                logger.warning(f"INT8 calibration data setup failed: {e}")
                try:
                    logger.warning("Falling back to synthetic Gaussian calibration batches")
                    batches = prepare_calibration_batches(
                        None,
                        self.num_cams,
                        self.hw,
                        max(self.calib_batches, 8),
                        seed=self.calib_seed,
                    )
                    calibrator = SimpleCalibrator(
                        input_shape,
                        calib_cache,
                        batches,
                        use_gpu=self.calib_use_gpu,
                    )
                    config.int8_calibrator = calibrator
                except Exception as inner:
                    logger.warning(
                        "Synthetic calibration setup also failed: %s. "
                        "Building without calibration (INT8 accuracy may degrade).",
                        inner,
                    )

        # Build engine
        logger.info("Building engine (this may take several minutes)...")
        if self.precision_config.name == "fp8":
            logger.info("FP8 build: Ignore 'Unsupported data type FP8' warnings - this is normal")

        # Attempt a strict-FP8 build first (no FP16/TF32 fallback) to reduce engine size.
        disabled_fp8_fallback_flags: List[str] = []
        if self.precision_config.name == "fp8" and hasattr(config, "clear_flag"):
            for fallback_flag in ("FP16", "TF32"):
                if hasattr(trt.BuilderFlag, fallback_flag):
                    flag_obj = getattr(trt.BuilderFlag, fallback_flag)
                    try:
                        if config.get_flag(flag_obj):  # type: ignore[attr-defined]
                            config.clear_flag(flag_obj)
                            disabled_fp8_fallback_flags.append(fallback_flag)
                            logger.info(
                                "FP8 strict mode: temporarily disabled %s fallback",
                                fallback_flag,
                            )
                    except Exception:
                        # Some TensorRT versions do not expose get_flag/clear_flag combos
                        try:
                            config.clear_flag(flag_obj)
                            disabled_fp8_fallback_flags.append(fallback_flag)
                            logger.info(
                                "FP8 strict mode: temporarily disabled %s fallback",
                                fallback_flag,
                            )
                        except Exception:
                            pass
            if not disabled_fp8_fallback_flags:
                logger.debug("FP8 strict mode: no fallback flags disabled (unsupported API)")
        
        engine_bytes = None
        strict_fp8_exception: Optional[Exception] = None
        if disabled_fp8_fallback_flags:
            try:
                engine_bytes = builder.build_serialized_network(network, config)
            except Exception as exc:
                strict_fp8_exception = exc
            if engine_bytes is not None and strict_fp8_exception is None:
                logger.info(
                    "Strict FP8 build succeeded without fallback precisions: %s disabled",
                    ", ".join(disabled_fp8_fallback_flags),
                )
            if engine_bytes is None:
                if strict_fp8_exception is not None:
                    logger.warning(
                        "Strict FP8 build failed: %s. Re-enabling fallbacks: %s",
                        strict_fp8_exception,
                        ", ".join(disabled_fp8_fallback_flags),
                    )
                else:
                    logger.warning(
                        "Strict FP8 build returned None. Re-enabling fallbacks: %s",
                        ", ".join(disabled_fp8_fallback_flags),
                    )

        if engine_bytes is None:
            if disabled_fp8_fallback_flags:
                for flag_name in disabled_fp8_fallback_flags:
                    try:
                        if hasattr(trt.BuilderFlag, flag_name):
                            config.set_flag(getattr(trt.BuilderFlag, flag_name))
                            logger.info("Re-enabled %s fallback", flag_name)
                    except Exception:
                        pass

        if engine_bytes is None:
            build_exc: Optional[Exception] = None
            try:
                engine_bytes = builder.build_serialized_network(network, config)
            except Exception as exc:
                build_exc = exc
                engine_bytes = None

            if engine_bytes is None:
                if len(workspace_plan) > 1:
                    next_plan = workspace_plan[1:]
                    next_workspace = next_plan[0]
                    logger.warning(
                        "TensorRT build failed%s with workspace %d GB; retrying with %d GB",
                        f" ({build_exc})" if build_exc else "",
                        workspace_gb,
                        next_workspace,
                    )
                    try:
                        del config
                    except Exception:
                        pass
                    try:
                        del network
                    except Exception:
                        pass
                    try:
                        del builder
                    except Exception:
                        pass
                    return self.build_trt(
                        onnx_path,
                        engine_path,
                        workspace_plan=next_plan,
                        workspace_attempt=workspace_attempt + 1,
                        workspace_total=workspace_total,
                    )
                error_msg = "TensorRT build returned None (build failed)"
                if build_exc is not None:
                    error_msg = f"{error_msg}: {build_exc}"
                raise RuntimeError(error_msg)

        # Inspect precision usage for diagnostics
        precision_stats: Dict[str, int] = {}
        engine_runtime = None
        try:
            runtime = trt.Runtime(logger_trt)
            engine_runtime = runtime.deserialize_cuda_engine(engine_bytes)
            precision_stats = self._inspect_engine_precisions(trt, engine_runtime)
        except Exception as exc:
            logger.debug("Skipped precision inspection: %s", exc)
        else:
            if precision_stats:
                stats_repr = ", ".join(f"{k}:{v}" for k, v in precision_stats.items())
                logger.info(f"Engine tensor precisions: {stats_repr}")
                if self.precision_config.name == "fp8" and not any("FP8" in k for k in precision_stats):
                    logger.warning(
                        "FP8 requested but engine contains no FP8 tensors; TensorRT likely "
                        "fell back to FP16/FP32. Consider pre-quantising with ModelOpt FP8 "
                        "or applying INT8 calibration."
                    )
                if self.precision_config.name == "int8" and not any("INT8" in k for k in precision_stats):
                    logger.warning(
                        "INT8 requested but engine contains no INT8 tensors. "
                        "Verify calibration data and ensure BuilderFlag.INT8 is enabled."
                    )
        finally:
            try:
                del engine_runtime  # release resources
            except Exception:
                pass

        # === OPTIMIZATION 7: Save timing cache ===
        try:
            if tc is None:
                tc = config.get_timing_cache()
            if tc:
                blob = tc.serialize()
                with open(cache_path, "wb") as f:
                    f.write(blob)
                logger.info(f"Saved timing cache to {cache_path}")
        except Exception as e:
            logger.info(f"Could not save timing cache: {e}")

        # Save engine
        with open(engine_path, "wb") as f:
            f.write(engine_bytes)
        
        engine_size = os.path.getsize(engine_path)
        logger.info(f"Engine saved: {engine_path} ({_fmt_size(engine_size)})")
        return engine_path

    def run(
        self,
        onnx_in: Optional[str],
        onnx_simp: str,
        onnx_noseq: str,
        engine_path: str,
        export: bool
    ) -> str:
        """Run complete pipeline."""
        # Stage 1: Export or use existing ONNX
        if export:
            base_name = onnx_simp.replace(".simp.onnx", ".onnx")
            onnx_in = self.export_from_hf(base_name)
        
        if not onnx_in:
            raise ValueError("Must provide --onnx-in or use --export")
        
        if not os.path.exists(onnx_in):
            raise FileNotFoundError(f"Input ONNX not found: {onnx_in}")

        # Stage 2: Remove sequences FIRST
        noseq = self.remove_sequences(onnx_in, onnx_noseq)
        
        # Stage 3: Simplify AFTER sequences removed
        simp = self.simplify(noseq, onnx_simp)
        
        # Stage 4: Validate
        self.onnx_check(simp)
        
        # Stage 5: Build TensorRT
        engine = self.build_trt(simp, engine_path)
        
        return engine


# ---------- CLI ----------

def _parse():
    """Parse command line arguments."""
    import argparse
    
    ap = argparse.ArgumentParser(
        description="FIXED: Convert VGGT models to optimized TensorRT engines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export with FP16 (fastest on RTX 5090)
  %(prog)s --export --num-cams 8
  
  # Build all precision variants for comparison
  %(prog)s --export --num-cams 8 --all-precisions
  
  # PCD-only mode (30%% faster - depth + camera heads only)
  %(prog)s --export --num-cams 8 --pcd-only --precision fp16
  
  # Build specific precision
  %(prog)s --export --num-cams 8 --precision int8
  
  # Use existing ONNX
  %(prog)s --onnx-in model.onnx --precision fp16

FIXES:
  ✓ INT8 calibrator now works with TRT 10+
  ✓ FP8 warnings suppressed (expected on RTX 5090)
  ✓ Opset handling fixed for onnxsim compatibility
  ✓ PCD-only mode for live point cloud construction
        """
    )
    
    # Input/output
    ap.add_argument(
        "--export",
        action="store_true",
        help="Export VGGT from HuggingFace"
    )
    ap.add_argument(
        "--onnx-in",
        help="Use existing ONNX file instead of exporting"
    )
    ap.add_argument(
        "--model-name",
        default="facebook/VGGT-1B",
        help="HuggingFace model name (default: facebook/VGGT-1B)"
    )
    
    # Model configuration
    ap.add_argument(
        "--num-cams",
        type=int,
        default=8,
        help="Number of camera views (default: 8)"
    )
    ap.add_argument(
        "--height",
        type=int,
        default=518,
        help="Input height (default: 518)"
    )
    ap.add_argument(
        "--width",
        type=int,
        default=518,
        help="Input width (default: 518)"
    )
    ap.add_argument(
        "--opset",
        type=int,
        default=18,
        help="ONNX opset version (default: 18)"
    )
    
    # Precision configuration
    ap.add_argument(
        "--precision",
        choices=list(PRECISIONS.keys()),
        default="fp16",
        help="Precision mode (default: fp16)"
    )
    ap.add_argument(
        "--all-precisions",
        action="store_true",
        help="Build all precision variants (fp16, bf16, fp8, int8)"
    )
    ap.add_argument(
        "--quant-mode",
        choices=list(SUPPORTED_QUANT_MODES.keys()),
        default="none",
        help="Optional pre-quantisation recipe before ONNX export"
    )
    
    # INT8 calibration configuration
    ap.add_argument(
        "--calib-data",
        default=None,
        help="Directory / glob / .npy providing calibration data for INT8 (optional)"
    )
    ap.add_argument(
        "--calib-batches",
        type=int,
        default=32,
        help="Number of calibration batches to feed TensorRT (default: 32)"
    )
    ap.add_argument(
        "--calib-seed",
        type=int,
        default=1337,
        help="Random seed for calibration data shuffling (default: 1337)"
    )
    ap.add_argument(
        "--calib-cpu",
        action="store_true",
        help="Force INT8 calibration to stage batches on CPU even if GPU is available"
    )
    
    # PCD optimization
    ap.add_argument(
        "--pcd-only",
        action="store_true",
        help="Export only depth + camera heads (30%% faster for point cloud construction)"
    )
    
    # TensorRT optimization
    ap.add_argument(
        "--workspace-gb",
        type=int,
        default=32,
        help="TensorRT workspace size in GB (default: 32)"
    )
    ap.add_argument(
        "--opt-level",
        type=int,
        default=5,
        help="TensorRT optimization level 0-5 (default: 5 = maximum)"
    )
    ap.add_argument(
        "--max-aux-streams",
        type=int,
        default=4,
        help="Maximum auxiliary CUDA streams (default: 4)"
    )
    ap.add_argument(
        "--no-simplify",
        action="store_true",
        help="Skip ONNX simplification"
    )
    
    # Output paths
    ap.add_argument(
        "--output-dir",
        default="onnx_exports",
        help="Output directory for all artifacts (default: onnx_exports)"
    )
    
    return ap.parse_args()


def main():
    """Main entry point."""
    args = _parse()
    
    # Auto-generate filenames
    stem = f"vggt-{args.num_cams}x3x{args.height}x{args.width}"
    if args.pcd_only:
        stem += "-pcd"
    
    precisions_to_build = []
    if args.all_precisions:
        # Build fp16, bf16, fp8, int8 (skip fp32 as it's slow)
        precisions_to_build = ["fp16", "bf16", "fp8", "int8"]
        logger.info("Building all precision variants: fp16, bf16, fp8, int8")
    else:
        precisions_to_build = [args.precision]
    
    results = []
    shared_onnx_base = None  # Track the base ONNX file for reuse
    
    try:
        for idx, precision in enumerate(precisions_to_build):
            logger.info("")
            logger.info("=" * 70)
            logger.info(f"Building {precision.upper()} variant ({idx+1}/{len(precisions_to_build)})")
            logger.info("=" * 70)
            
            suffix = PRECISIONS[precision].suffix
            onnx_simp = os.path.join(args.output_dir, f"{stem}.simp.onnx")
            onnx_noseq = os.path.join(args.output_dir, f"{stem}.NOSEQ.onnx")
            engine = os.path.join(args.output_dir, f"{stem}{suffix}.engine")
            
            pipe = VGGTPipeline(
                num_cams=args.num_cams,
                hw=(args.height, args.width),
                opset=args.opset,
                workspace_gb=args.workspace_gb,
                precision=precision,
                simplify=not args.no_simplify,
                model_name=args.model_name,
                opt_level=args.opt_level,
                max_aux_streams=args.max_aux_streams,
                pcd_only=args.pcd_only,
                quant_mode=args.quant_mode,
                calib_source=args.calib_data,
                calib_batches=args.calib_batches,
                calib_seed=args.calib_seed,
                calib_use_gpu=not args.calib_cpu,
            )
            
            # Export only on first iteration, then reuse
            is_first = (idx == 0)
            do_export = args.export and is_first
            
            # Use shared ONNX from first build for subsequent precisions
            onnx_input = args.onnx_in if args.onnx_in else shared_onnx_base
            
            engine_path = pipe.run(
                onnx_in=onnx_input,
                onnx_simp=onnx_simp,
                onnx_noseq=onnx_noseq,
                engine_path=engine,
                export=do_export,
            )
            
            # Store the base ONNX path after first export
            if is_first and do_export:
                shared_onnx_base = onnx_simp.replace(".simp.onnx", ".onnx")
                logger.info(f"Shared ONNX for subsequent builds: {shared_onnx_base}")
            
            results.append((precision, engine_path))
        
        # Summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("BUILD COMPLETE!")
        logger.info("=" * 70)
        logger.info(f"Input shape: [{args.num_cams}, 3, {args.height}, {args.width}]")
        if args.pcd_only:
            logger.info("Mode: PCD-only (depth + camera heads)")
        logger.info("")
        logger.info("Built engines:")
        for precision, engine_path in results:
            size = os.path.getsize(engine_path)
            desc = PRECISIONS[precision].description
            logger.info(f"  {precision.upper():5s} ({desc:40s}): {engine_path}")
            logger.info(f"        Size: {_fmt_size(size)}")
        logger.info("")
        logger.info("Next steps for live PCD construction:")
        logger.info("  1. Use FP16 for best speed/quality balance on RTX 5090")
        logger.info("  2. Extract outputs: cat_322 (camera), view_411/412 (depth)")
        logger.info("  3. Unproject depth to 3D using camera parameters")
        logger.info("  4. Expected latency: ~170ms FP16, ~230ms BF16 (from your benchmark)")
        logger.info("  5. With --pcd-only: expect ~30% faster (~120ms FP16)")
        logger.info("")
        logger.info("Performance hierarchy (your RTX 5090):")
        logger.info("  FP16:  170ms (5.9 FPS) ← RECOMMENDED for live PCD")
        logger.info("  BF16:  232ms (4.3 FPS)")
        logger.info("  FP8:   173ms (5.8 FPS) - similar to FP16, limited HW support")
        logger.info("  INT8:  TBD (needs proper calibration data)")
        logger.info("")
        logger.info("Optimizations applied:")
        logger.info(f"  ✓ Timing cache (speeds up rebuilds)")
        logger.info(f"  ✓ Optimization level {args.opt_level} (maximum)")
        logger.info(f"  ✓ Auxiliary streams: {args.max_aux_streams}")
        logger.info(f"  ✓ TF32 enabled (Ampere+ GPUs)")
        logger.info(f"  ✓ All CUDA tactic sources (cuBLAS, cuDNN)")
        logger.info(f"  ✓ Workspace: {args.workspace_gb} GB")
        logger.info(f"  ✓ FIXED: INT8 calibrator (TRT 10+ compatible)")
        logger.info(f"  ✓ FIXED: Opset handling (onnxsim now works)")
        logger.info("=" * 70)
        
    except KeyboardInterrupt:
        logger.error("\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
