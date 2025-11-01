"""TensorRT wrapper for VGGT engines (PCD package)."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import trt_utils

try:
    from onnx.tools.trt_inference import SimpleTrtRunner  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("onnx.tools.trt_inference.SimpleTrtRunner is required") from exc


LOGGER = logging.getLogger(__name__)


def _array_stats(arr: np.ndarray) -> Tuple[float, float, int, int]:
    """Return (min, max, positive_count, total) ignoring NaNs."""
    array = np.asarray(arr)
    total = int(array.size)
    if total == 0:
        return float("nan"), float("nan"), 0, 0
    finite = np.isfinite(array)
    if not np.any(finite):
        return float("nan"), float("nan"), 0, total
    finite_vals = array[finite]
    pos = finite_vals > 0
    return float(finite_vals.min()), float(finite_vals.max()), int(pos.sum()), total


def _orthonormalize_rotations(rot: np.ndarray, atol: float = 1e-3) -> np.ndarray:
    """Project rotation matrices back onto SO(3) if they drift numerically."""
    matrices = np.asarray(rot, dtype=np.float32)
    if matrices.ndim < 3 or matrices.shape[-2:] != (3, 3):
        raise ValueError(f"Expected (...,3,3) rotations; got {matrices.shape}")
    reshaped = matrices.reshape(-1, 3, 3)
    corrected = reshaped.copy()
    needs_fix = []
    for idx, mat in enumerate(reshaped):
        err = float(np.linalg.norm(mat.T @ mat - np.eye(3, dtype=np.float32), ord="fro"))
        if err > atol or not np.isfinite(err):
            u, _, vh = np.linalg.svd(mat)
            candidate = u @ vh
            if np.linalg.det(candidate) < 0:
                u[:, -1] *= -1.0
                candidate = u @ vh
            corrected[idx] = candidate.astype(np.float32)
            needs_fix.append(err)
    if needs_fix:
        LOGGER.debug(
            "Re-orthonormalized %d rotation matrices (max deviation %.2e).",
            len(needs_fix),
            max(needs_fix),
        )
    return corrected.reshape(matrices.shape)


def _collapse_repeated_segments(flat: np.ndarray, bases: Iterable[int]) -> Tuple[np.ndarray, int]:
    """Collapse repeated pose encodings when TensorRT tiles the last dimension."""
    last_dim = flat.shape[-1]
    for base in bases:
        if base <= 0 or last_dim % base != 0:
            continue
        groups = last_dim // base
        if groups == 1:
            return flat, last_dim
        reshaped = flat.reshape(flat.shape[0], flat.shape[1], groups, base)
        first = reshaped[..., 0, :]
        max_dev = float(np.max(np.abs(reshaped - first[..., None, :])))
        if not np.isfinite(max_dev):
            continue
        if max_dev < 1e-4:
            LOGGER.debug(
                "Pose tensor appears repeated (%d segments of %d). Collapsing to the first slice.",
                groups,
                base,
            )
            return first, base
    return flat, last_dim


def _prepare_batch(images: Sequence[np.ndarray], norm: str = "imagenet") -> np.ndarray:
    if len(images) == 0:
        raise ValueError("At least one image is required")
    shapes = {img.shape for img in images}
    if len(shapes) != 1:
        raise ValueError("All input images must share the same shape")
    batch = np.stack(images, axis=0).astype(np.float32)
    # To CHW
    batch = np.transpose(batch, (0, 3, 1, 2))
    # Apply normalization
    if norm == "imagenet":
        # ImageNet normalization (mean/std)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        batch = (batch - mean) / std
    elif norm in {"zero_center", "minus_one_to_one", "tanh"}:
        # Map [0,1] -> [-1,1]
        batch = (batch - 0.5) / 0.5
    else:
        LOGGER.warning("Unknown normalization '%s'; using identity.", norm)
    return batch


def _quat_to_matrix_np(quat: np.ndarray) -> np.ndarray:
    """Convert XYZW quaternions to rotation matrices."""
    q = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    norm = np.where(norm > 1e-8, norm, 1.0)
    q = q / norm
    x, y, z, w = np.moveaxis(q, -1, 0)
    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w
    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w

    rot = np.stack(
        [
            1 - 2 * (yy + zz),
            2 * (xy - zw),
            2 * (xz + yw),
            2 * (xy + zw),
            1 - 2 * (xx + zz),
            2 * (yz - xw),
            2 * (xz - yw),
            2 * (yz + xw),
            1 - 2 * (xx + yy),
        ],
        axis=-1,
    )
    rot = rot.reshape(q.shape[:-1] + (3, 3))
    return rot.astype(np.float32)


def _decode_pose_from_encoding(
    encoding: np.ndarray,
    image_hw: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode 9-D VGGT pose encoding (T, quat, FoV) to extrinsics / intrinsics."""
    if encoding.shape[-1] != 9:
        raise ValueError(f"Expected pose encoding width 9, got {encoding.shape[-1]}")
    trans = encoding[..., :3]
    quat = encoding[..., 3:7]
    fov_h = encoding[..., 7]
    fov_w = encoding[..., 8]

    rot = _quat_to_matrix_np(quat)
    rot = _orthonormalize_rotations(rot)
    extr = np.concatenate([rot, trans[..., None]], axis=-1)

    H, W = image_hw
    eps = 1e-6
    tan_fov_h = np.tan(np.clip(fov_h / 2.0, eps, np.pi / 2 - eps))
    tan_fov_w = np.tan(np.clip(fov_w / 2.0, eps, np.pi / 2 - eps))
    fy = (H / 2.0) / tan_fov_h
    fx = (W / 2.0) / tan_fov_w
    intr = np.zeros(encoding.shape[:-1] + (3, 3), dtype=np.float32)
    intr[..., 0, 0] = fx.astype(np.float32)
    intr[..., 1, 1] = fy.astype(np.float32)
    intr[..., 0, 2] = W / 2.0
    intr[..., 1, 2] = H / 2.0
    intr[..., 2, 2] = 1.0
    return extr.astype(np.float32), intr


def _decode_pose_tensor(
    pose_array: np.ndarray,
    image_hw: Tuple[int, int],
    tensor_name: str,
    pose_decoder: Optional[Callable[[torch.Tensor, Tuple[int, int]], Tuple[torch.Tensor, Optional[torch.Tensor]]]],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Decode TensorRT camera outputs into extrinsics/intrinsics."""
    arr = np.asarray(pose_array, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("Pose tensor is empty.")
    if arr.ndim == 1:
        arr = arr.reshape(1, 1, -1)
    elif arr.ndim == 2:
        arr = arr.reshape(arr.shape[0], 1, arr.shape[1])
    elif arr.ndim >= 4 and arr.shape[-2:] == (3, 4):
        extr = arr.astype(np.float32)
        extr[..., :3] = _orthonormalize_rotations(extr[..., :3])
        return extr, None
    if arr.ndim > 3:
        arr = arr.reshape(arr.shape[0], arr.shape[1], -1)
    flat = arr.reshape(arr.shape[0], arr.shape[1], -1)
    flat, last_dim = _collapse_repeated_segments(flat, bases=(9, 16, 12, 7, 4))
    if last_dim > 9 and last_dim % 9 == 0:
        segments = last_dim // 9
        LOGGER.info(
            "Pose tensor '%s' appears to contain %d stacked 9-D segments; using the first segment.",
            tensor_name,
            segments,
        )
        flat = flat.reshape(flat.shape[0], flat.shape[1], segments, 9)[..., 0, :]
        last_dim = 9
    LOGGER.info(
        "Pose tensor '%s' flattened to (..., %d) from raw shape %s.",
        tensor_name,
        last_dim,
        tuple(pose_array.shape),
    )
    if last_dim <= 32:
        sample = np.array2string(
            flat.reshape(-1, last_dim)[0],
            precision=4,
            floatmode="fixed",
            suppress_small=True,
            max_line_width=160,
        )
        LOGGER.info("Sample values %s", sample)

    if last_dim == 9:
        extr_np, intr_np = _decode_pose_from_encoding(flat, image_hw)
        return extr_np, intr_np
    if last_dim == 7:
        quat = flat[..., :4]
        trans = flat[..., 4:7]
        quat_t = torch.from_numpy(quat)
        rot = quat_to_mat(quat_t).detach().cpu().numpy().astype(np.float32)
        rot = _orthonormalize_rotations(rot)
        extr = np.concatenate([rot, trans[..., None]], axis=-1).astype(np.float32)
        return extr, None
    if last_dim == 12:
        extr = flat.reshape(flat.shape[0], flat.shape[1], 3, 4).astype(np.float32)
        extr[..., :3, :3] = _orthonormalize_rotations(extr[..., :3, :3])
        return extr, None
    if last_dim == 16:
        mats = flat.reshape(flat.shape[0], flat.shape[1], 4, 4)
        rot = _orthonormalize_rotations(mats[..., :3, :3])
        trans = mats[..., :3, 3]
        extr = np.concatenate([rot, trans[..., None]], axis=-1).astype(np.float32)
        return extr, None
    raise ValueError(f"Unsupported pose tensor width {last_dim} from '{tensor_name}'.")


@dataclass
class VGGTOutput:
    depth_maps: List[np.ndarray]
    raw_outputs: List[np.ndarray]
    camera_extrinsics: Optional[np.ndarray] = None
    camera_intrinsics: Optional[np.ndarray] = None


class TRTVGGT:
    def __init__(self, engine: Path, verbose: bool = False, input_normalization: str = "zero_center") -> None:
        self.engine_path = Path(engine)
        if not self.engine_path.exists():
            raise FileNotFoundError(self.engine_path)
        self.runner = SimpleTrtRunner(str(self.engine_path), verbose=verbose, force_sync=False)
        self.input_normalization = input_normalization
        LOGGER.info("Loaded VGGT engine %s", self.engine_path)

    @classmethod
    def from_directory(cls, engine_dir: Path, precision: str = "auto", base_name: str = "vggt", **kwargs) -> "TRTVGGT":
        engines = trt_utils.discover_engines(engine_dir, None, base_name)
        engine = trt_utils.select_engine(engines, precision)
        return cls(engine=engine, **kwargs)

    def _log_outputs(self, outputs: Sequence[np.ndarray]) -> None:
        metas = list(self.runner.output_meta)
        for meta, array in zip(metas, outputs):
            arr = np.asarray(array)
            stats = _array_stats(arr)
            LOGGER.info(
                "VGGT output '%s': binding_shape=%s array_shape=%s dtype=%s min=%.4f max=%.4f pos=%d/%d",
                meta.get("name"),
                tuple(meta.get("shape", ())),
                tuple(arr.shape),
                arr.dtype,
                stats[0],
                stats[1],
                stats[2],
                stats[3],
            )

    def _extract_depths(self, outputs: List[np.ndarray]) -> List[np.ndarray]:
        """Heuristically extract per-view depth maps from engine outputs."""
        depth_maps: List[np.ndarray] = []
        # First pass: prefer outputs with 'depth' in the name
        metas = list(self.runner.output_meta)
        ranked = list(range(len(outputs)))
        ranked.sort(key=lambda i: ("depth" not in metas[i]["name"], metas[i]["name"]))

        for idx in ranked:
            meta = metas[idx]
            array = outputs[idx]
            name = meta["name"]
            shape = tuple(array.shape)
            LOGGER.debug("VGGT output '%s' shape=%s", name, shape)
            arr = array
            # Flatten potential (B,V,...) to (N,...)
            if arr.ndim >= 5 and ("view" in name or arr.shape[1] in {2, 4, 8, 16}):
                arr = arr.reshape(arr.shape[0] * arr.shape[1], *arr.shape[2:])
            # Channel squeeze if single-channel
            if arr.ndim == 4:
                if arr.shape[1] in {1, 2}:
                    arr = arr[:, 0, :, :]
                elif arr.shape[-1] in {1, 2}:
                    arr = arr[:, :, :, 0]
                elif arr.shape[1] == 3 or arr.shape[-1] == 3:
                    # Likely point map; skip
                    continue
            # Accept only (N,H,W)
            if arr.ndim != 3:
                continue
            # Basic plausibility: spatial dims reasonably large
            H, W = arr.shape[-2], arr.shape[-1]
            if min(H, W) < 64:
                continue
            # Split into per-view
            for n in range(arr.shape[0]):
                d = np.asarray(arr[n], dtype=np.float32)
                finite = np.isfinite(d)
                pos = finite & (d > 0)
                mn = float(np.nanmin(d)) if np.any(finite) else float("nan")
                mx = float(np.nanmax(d)) if np.any(finite) else float("nan")
                cnt = int(np.count_nonzero(pos))
                LOGGER.debug("Candidate depth from '%s'[%d]: min=%.4f max=%.4f pos=%d/%d", name, n, mn, mx, cnt, d.size)
                # Keep candidates; final selection is done by caller
                depth_maps.append(d)
        return depth_maps

    def _decode_camera_params(
        self,
        outputs: List[np.ndarray],
        image_hw: Tuple[int, int],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # type: ignore
        except ImportError:  # pragma: no cover
            LOGGER.warning("pose_encoding_to_extri_intri not available; skipping camera decode.")
            pose_encoding_to_extri_intri = None  # type: ignore

        metas = list(self.runner.output_meta)
        decode_errors: List[str] = []
        for meta, arr in zip(metas, outputs):
            name = meta["name"]
            pose_array = np.asarray(arr, dtype=np.float32)
            LOGGER.debug("Attempting camera decode from '%s' with shape %s", name, pose_array.shape)
            try:
                extrinsic, intrinsic = _decode_pose_tensor(
                    pose_array,
                    image_hw,
                    tensor_name=name,
                    pose_decoder=pose_encoding_to_extri_intri,
                )
            except ValueError as exc:
                decode_errors.append(f"{name}: {exc}")
                continue
            except Exception as exc:  # pragma: no cover - diagnostic
                decode_errors.append(f"{name}: {exc}")
                continue
            LOGGER.info("Decoded camera parameters from TensorRT output '%s'.", name)
            return extrinsic, intrinsic
        if decode_errors:
            LOGGER.warning("Pose decoding attempts failed: %s", "; ".join(decode_errors))
        return None, None

    def run(
        self,
        images: Sequence[np.ndarray],
        norm: Optional[str] = None,
        retry_norms: Sequence[str] = (),
    ) -> VGGTOutput:
        norm_sequence: List[str] = []
        primary_norm = norm or self.input_normalization
        norm_sequence.append(primary_norm)
        for candidate in retry_norms:
            if candidate not in norm_sequence:
                norm_sequence.append(candidate)

        tried: List[str] = []
        for current_norm in norm_sequence:
            tried.append(current_norm)
            LOGGER.debug("Running TensorRT VGGT with '%s' normalization.", current_norm)
            batch = _prepare_batch(images, norm=current_norm)
            outputs = self.runner.infer(batch, copy_outputs=True)
            self._log_outputs(outputs)
            depth_maps = self._extract_depths(outputs)
            has_positive = any(np.isfinite(dm).any() and (dm > 0).any() for dm in depth_maps)
            if not has_positive:
                LOGGER.warning(
                    "VGGT depths invalid (all <=0) with '%s' normalization.",
                    current_norm,
                )
                continue
            LOGGER.info("VGGT depths found using '%s' normalization.", current_norm)
            camera_extrinsics, camera_intrinsics = self._decode_camera_params(
                outputs,
                (int(images[0].shape[0]), int(images[0].shape[1])),
            )
            return VGGTOutput(
                depth_maps=depth_maps,
                raw_outputs=outputs,
                camera_extrinsics=camera_extrinsics,
                camera_intrinsics=camera_intrinsics,
            )
        raise RuntimeError(
            "VGGT produced no positive depth values. Normalizations tried: {}".format(", ".join(tried))
        )
