"""Live point-cloud reconstruction pipeline entrypoint.

The script supports a two stage workflow:

* **Test mode**: bootstrap a metric map by running a VGGT TensorRT engine on
  the first 8 images.  The resulting point cloud (or TSDF) is persisted.
* **Live updates**: subsequent frames are processed with a Depth Anything v2
  TensorRT engine.  Each relative depth map is aligned to the current map,
  back-projected and fused.

The implementation keeps the heavy lifting in helper modules under
``reconstruction/`` and orchestrates them here.  All code is additive so existing
scripts remain untouched.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - OpenCV optional
    cv2 = None

from .pcd import depth_anything, fusion, io_deepview, io_utils, metric_depth, raycast, vggt_trt


LOGGER = logging.getLogger("pcd")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Live point-cloud reconstruction using VGGT + Depth Anything.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="generic",
        choices=["generic", "deepview"],
        help="Dataset adapter to use (generic directory layout or DeepView light-field).",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Root directory containing scenes for the selected dataset.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Scene identifier within --dataset-root (DeepView or generic).",
    )
    parser.add_argument(
        "--n-views",
        type=int,
        default=8,
        help="Number of camera views to use for VGGT bootstrap and live updates.",
    )
    parser.add_argument(
        "--view-select",
        type=str,
        default="max-spread",
        choices=["max-spread", "front-arc", "random"],
        help="Automatic view selection strategy when --views is not provided.",
    )
    parser.add_argument(
        "--views",
        type=str,
        default=None,
        help="Comma separated list of camera IDs to use (overrides auto selection).",
    )
    parser.add_argument(
        "--undistort",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="Fisheye undistortion mode (auto enables for fisheye cameras only).",
    )
    parser.add_argument(
        "--rectify-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(518, 518),
        help="Target resolution for rectified images (VGGT expects 518x518).",
    )
    parser.add_argument(
        "--cache-rectify-maps",
        action="store_true",
        help="Persist fisheye rectification maps to disk for reuse.",
    )
    parser.add_argument(
        "--validate-projection",
        action="store_true",
        help="Project the DeepView README reference world point and report pixels.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Initial frame index to bootstrap from (DeepView/video datasets).",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Process every Nth frame when iterating through sequences.",
    )
    parser.add_argument(
        "--timestamp",
        type=float,
        default=None,
        help="Optional timestamp hint (reserved for future synchronization hooks).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="images",
        choices=["images", "videos", "webcams"],
        help="Legacy generic dataset input modality (used when --dataset=generic).",
    )
    parser.add_argument(
        "--images",
        type=str,
        default=None,
        help="Directory of per-camera images (used for generic datasets).",
    )
    parser.add_argument(
        "--images_live",
        type=str,
        default=None,
        help="Optional separate directory for live frames in image mode.",
    )
    parser.add_argument(
        "--videos",
        type=str,
        default=None,
        help="Directory containing per-camera videos (generic mode).",
    )
    parser.add_argument(
        "--webcams",
        type=str,
        nargs="*",
        help="List of webcam indices or name=index pairs (generic live capture).",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Optional limit on the number of live frame batches to process.",
    )
    parser.add_argument(
        "--intrinsics",
        type=str,
        default=None,
        help="Camera intrinsics (JSON/YAML/NPZ) for generic datasets.",
    )
    parser.add_argument(
        "--poses",
        type=str,
        default=None,
        help="Camera-to-world poses (JSON/YAML/NPZ) for generic datasets.",
    )
    parser.add_argument(
        "--initial_map",
        type=str,
        default=None,
        help="Existing map (.ply or .npz). If not provided and --test-mode=1 bootstrap from VGGT.",
    )
    parser.add_argument(
        "--vggt_engine",
        type=str,
        default=None,
        help="VGGT TensorRT engine used for bootstrap.",
    )
    parser.add_argument(
        "--vggt-norm",
        type=str,
        default="zero_center",
        choices=["imagenet", "zero_center", "minus_one_to_one", "tanh"],
        help="Image normalization applied before VGGT TensorRT inference.",
    )
    parser.add_argument(
        "--vggt-backend",
        type=str,
        default="trt",
        choices=["trt", "huggingface"],
        help="Implementation used for VGGT bootstrap (TensorRT engine or HuggingFace PyTorch).",
    )
    parser.add_argument(
        "--hf-checkpoint",
        type=str,
        default="facebook/VGGT-1B",
        help="Hugging Face repo id for VGGT weights when --vggt-backend=huggingface.",
    )
    parser.add_argument(
        "--hf-weights",
        type=str,
        default=None,
        help="Optional local path or URL to VGGT weights (.pt) when using the HuggingFace backend.",
    )
    parser.add_argument(
        "--hf-device",
        type=str,
        default=None,
        help="Optional device override for HuggingFace VGGT (default: auto-select).",
    )
    parser.add_argument(
        "--hf-fallback",
        action="store_true",
        help="When set, allow HuggingFace VGGT fallback to recover cameras if the TensorRT engine omits them.",
    )
    parser.add_argument(
        "--use-dataset-cameras",
        action="store_true",
        help="Use dataset-supplied intrinsics/extrinsics when available (default: disabled).",
    )
    parser.add_argument(
        "--pose-source",
        type=str,
        default="auto",
        choices=["auto", "dataset", "vggt"],
        help="Camera pose source during VGGT bootstrap: 'auto' prefers engine outputs, 'dataset' keeps GT poses, 'vggt' requires engine predictions.",
    )
    parser.add_argument(
        "--log-depth-stats",
        action="store_true",
        help="Log min/max depth statistics for each bootstrap view.",
    )
    parser.add_argument(
        "--debug-geo",
        action="store_true",
        help="Run geometry diagnostics (reprojection checks, epipolar consistency) after VGGT bootstrap.",
    )
    parser.add_argument(
        "--depth_engine_dir",
        type=str,
        default="onnx_exports/depth_anything",
        help="Directory containing Depth Anything TensorRT engines.",
    )
    parser.add_argument(
        "--depth_engine",
        type=str,
        default=None,
        help="Explicit path to a Depth Anything TensorRT engine (overrides --depth_engine_dir discovery).",
    )
    parser.add_argument(
        "--depth_workers",
        type=int,
        default=None,
        help="Number of Depth Anything workers/engines to spawn (default: n-views).",
    )
    parser.add_argument(
        "--trt_precision",
        type=str,
        default="auto",
        choices=["auto", "fp32", "fp16", "bf16", "fp8", "int8"],
        help="Preferred TensorRT precision.",
    )
    parser.add_argument(
        "--fusion",
        type=str,
        default="surfel",
        choices=["surfel", "tsdf"],
        help="Fusion back-end to use.",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.01,
        help="Voxel size (m) for TSDF or surfel radius proxy.",
    )
    parser.add_argument(
        "--trunc_margin",
        type=float,
        default=0.04,
        help="TSDF truncation margin (m).",
    )
    parser.add_argument(
        "--align_model",
        type=str,
        default="auto",
        choices=["auto", "scale", "scale_shift", "invdepth_scale_shift"],
        help="Alignment model selection strategy.",
    )
    parser.add_argument(
        "--smooth_scale",
        type=float,
        default=0.0,
        help="Legacy EMA smoothing factor (retained for compatibility; EKF-based smoothing is used instead).",
    )
    parser.add_argument(
        "--align_max_samples",
        type=int,
        default=20000,
        help="Maximum number of pixels sampled per frame for metric alignment.",
    )
    parser.add_argument(
        "--align_process_noise",
        type=float,
        nargs=2,
        metavar=("SCALE_NOISE", "SHIFT_NOISE"),
        default=(1e-4, 1e-4),
        help="Process noise for the per-camera EKF (scale, shift).",
    )
    parser.add_argument(
        "--align_measurement_floor",
        type=float,
        default=1e-4,
        help="Minimum measurement noise (avoids over-confident updates).",
    )
    parser.add_argument(
        "--align_tsdf_weight",
        type=float,
        default=4.0,
        help="Inflation factor applied to measurement noise when TSDF residuals are large.",
    )
    parser.add_argument(
        "--per_camera_calib",
        type=str,
        default=None,
        help="Optional JSON to persist per-camera scale/shift.",
    )
    parser.add_argument(
        "--save_per_frame_depth",
        type=int,
        default=0,
        help="When non-zero, save aligned metric depth per frame.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory to store outputs (map, metrics, aligned depth).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--test-mode",
        type=int,
        default=0,
        help="Enable VGGT bootstrap then simulated live updates when set to 1.",
    )
    # Legacy compatibility flags (suppressed in help but accepted).
    parser.add_argument(
        "--num_cams",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--frame_step",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--random_views",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    return parser


def _parse_view_list(views: Optional[str]) -> Optional[List[str]]:
    if views is None:
        return None
    parsed = [view.strip() for view in views.split(",") if view.strip()]
    return parsed or None


def _normalize_args(args: argparse.Namespace) -> None:
    """
    Harmonize legacy aliases and enforce defaults on the parsed arguments.
    """
    if getattr(args, "num_cams", None):
        args.n_views = int(args.num_cams)
    if getattr(args, "frame_step", None):
        args.frame_stride = int(args.frame_step)
    args.n_views = max(1, int(args.n_views))
    args.frame_stride = max(1, int(args.frame_stride))
    if hasattr(args, "vggt_backend") and isinstance(args.vggt_backend, str):
        args.vggt_backend = args.vggt_backend.lower()


def _scale_intrinsics_local(K: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    scaled = np.array(K, dtype=np.float32)
    scaled[0, 0] *= scale_x
    scaled[0, 2] *= scale_x
    scaled[1, 1] *= scale_y
    scaled[1, 2] *= scale_y
    return scaled


def _run_vggt_hf(
    frames: Sequence[io_utils.FrameData],
    target_hw: Tuple[int, int],
    checkpoint: str,
    weight_override: Optional[str],
    device_override: Optional[str],
) -> List[dict]:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyTorch is required for the HuggingFace VGGT backend.") from exc

    try:
        from vggt.models.vggt import VGGT  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("VGGT python modules not available; cannot use HuggingFace backend.") from exc
    try:
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # type: ignore
        from vggt.utils.geometry import unproject_depth_map_to_point_map  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("VGGT utility modules not available for pose/depth decoding.") from exc

    weight_path: Optional[Path] = None
    if weight_override:
        candidate = Path(weight_override)
        if candidate.exists():
            weight_path = candidate
        else:
            raise FileNotFoundError(f"HuggingFace weight override not found: {candidate}")
    else:
        local_candidates = [
            Path("vggt_wrapped.pt"),
            Path("model.pt"),
            Path("weights") / "model.pt",
        ]
        for candidate in local_candidates:
            if candidate.exists():
                weight_path = candidate
                break
    if weight_path is None:
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "huggingface_hub is required to download VGGT weights or provide --hf-weights."
            ) from exc
        try:
            weight_path = Path(hf_hub_download(repo_id=checkpoint, filename="model.pt"))
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Unable to download VGGT weights from Hugging Face. Provide --hf-weights pointing to a local .pt file."
            ) from exc

    device = device_override or ("cuda" if torch.cuda.is_available() else "cpu")
    if device != "cpu" and not torch.cuda.is_available():
        LOGGER.warning("Requested HuggingFace VGGT device '%s' unavailable; falling back to CPU.", device)
        device = "cpu"

    LOGGER.info("Loading HuggingFace VGGT weights from %s", weight_path)
    state_dict = torch.load(str(weight_path), map_location="cpu")
    model = VGGT()
    missing = model.load_state_dict(state_dict, strict=False)
    if getattr(missing, "missing_keys", None):
        LOGGER.debug("VGGT missing keys: %s", missing.missing_keys)
    if getattr(missing, "unexpected_keys", None):
        LOGGER.debug("VGGT unexpected keys: %s", missing.unexpected_keys)
    model.eval()
    model = model.to(device)

    if device == "cuda":
        cap_major = torch.cuda.get_device_capability()[0]
        dtype = torch.bfloat16 if cap_major >= 8 else torch.float16
        autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)
    else:
        autocast_ctx = contextlib.nullcontext()

    images_np = np.stack([frame.image for frame in frames], axis=0).astype(np.float32)
    images_tensor = torch.from_numpy(images_np).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32)
    h_target, w_target = target_hw
    images_tensor = F.interpolate(images_tensor, size=(h_target, w_target), mode="bilinear", align_corners=False)
    images_tensor = images_tensor.unsqueeze(0)

    with torch.no_grad():
        with autocast_ctx:
            aggregated_tokens_list, ps_idx = model.aggregator(images_tensor)
        # Run heads in full precision for numerical stability.
        with torch.cuda.amp.autocast(enabled=False):
            pose_enc = model.camera_head(aggregated_tokens_list)[-1]
            depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images=images_tensor, patch_start_idx=ps_idx)
            point_map, point_conf = model.point_head(aggregated_tokens_list, images=images_tensor, patch_start_idx=ps_idx)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images_tensor.shape[-2:])
    extrinsic_np = extrinsic.detach().cpu().numpy()
    intrinsic_np = intrinsic.detach().cpu().numpy()
    depth_np = depth_map.detach().cpu().numpy()[0]  # (S,H,W)
    point_np = point_map.detach().cpu().numpy()[0]  # (S,H,W,3)
    depth_conf_np = depth_conf.detach().float().cpu().numpy()[0]  # (S,H,W)
    point_conf_np = point_conf.detach().float().cpu().numpy()[0]  # (S,H,W)
    point_unproj_np = unproject_depth_map_to_point_map(depth_np, extrinsic_np[0], intrinsic_np[0])  # (S,H,W,3)

    results: List[dict] = []
    for idx, frame in enumerate(frames):
        depth_i = depth_np[idx]
        world_points = point_unproj_np[idx]
        point_head_world = point_np[idx]
        mask = np.isfinite(depth_i) & (depth_i > 0)
        rgb = frame.image
        results.append(
            {
                "camera": frame.camera_id,
                "depth": depth_i.astype(np.float32),
                "depth_conf": depth_conf_np[idx].astype(np.float32),
                "intrinsic": intrinsic_np[0, idx].astype(np.float32),
                "extrinsic": extrinsic_np[0, idx].astype(np.float32),
                "point_map": point_head_world.astype(np.float32),
                "point_map_unprojected": world_points.astype(np.float32),
                "mask": mask,
                "rgb": rgb.astype(np.float32),
            }
        )
    return results


def _resize_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) is required for image resizing.")
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def _points_and_colors_from_depth(
    depth: np.ndarray,
    image: np.ndarray,
    intrinsics: np.ndarray,
    pose_wc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Back-project a depth map to 3D world points alongside per-point RGB colors.
    """
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("Depth map must be 2D for color extraction.")
    image_rgb = np.asarray(image, dtype=np.float32)
    if image_rgb.shape[:2] != depth.shape:
        raise ValueError("Image and depth dims must match for color extraction.")
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth.reshape(-1)
    valid = np.isfinite(z) & (z > 0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
    x = ((u.reshape(-1) - cx) / fx) * z
    y = ((v.reshape(-1) - cy) / fy) * z
    points_c = np.stack([x, y, z], axis=1)[valid]
    homog = np.concatenate([points_c, np.ones((points_c.shape[0], 1), dtype=np.float32)], axis=1)
    points_w = (pose_wc @ homog.T).T[:, :3]
    colors = (image_rgb.reshape(-1, 3)[valid] * 255.0).clip(0, 255).astype(np.uint8)
    return points_w.astype(np.float32), colors


def _bilinear_sample(depth: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinear sample depth map at fractional pixel coordinates."""
    h, w = depth.shape
    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    u1 = u0 + 1
    v1 = v0 + 1
    inside = (u0 >= 0) & (v0 >= 0) & (u1 < w) & (v1 < h)
    result = np.full(u.shape, np.nan, dtype=np.float32)
    if not np.any(inside):
        return result
    idx = np.where(inside)[0]
    u0_valid = u0[inside]
    v0_valid = v0[inside]
    u1_valid = u1[inside]
    v1_valid = v1[inside]
    d00 = depth[v0_valid, u0_valid]
    d01 = depth[v0_valid, u1_valid]
    d10 = depth[v1_valid, u0_valid]
    d11 = depth[v1_valid, u1_valid]
    finite = np.isfinite(d00) & np.isfinite(d01) & np.isfinite(d10) & np.isfinite(d11)
    if not np.any(finite):
        return result
    idx = idx[finite]
    u0_valid = u0_valid[finite]
    v0_valid = v0_valid[finite]
    d00 = d00[finite]
    d01 = d01[finite]
    d10 = d10[finite]
    d11 = d11[finite]
    fu = u[idx] - u0_valid.astype(np.float32)
    fv = v[idx] - v0_valid.astype(np.float32)
    result[idx] = (
        d00 * (1 - fu) * (1 - fv)
        + d01 * fu * (1 - fv)
        + d10 * (1 - fu) * fv
        + d11 * fu * fv
    ).astype(np.float32)
    return result


def _geometry_debug_report(
    depth_maps: Sequence[np.ndarray],
    frames: Sequence[io_utils.FrameData],
    intrinsics: Sequence[np.ndarray],
    poses_wc: Sequence[np.ndarray],
    sample_count: int = 2048,
) -> None:
    """Emit quick geometry diagnostics for reconstructed cameras."""
    rng = np.random.default_rng(12345)
    per_view_logs: List[str] = []
    packed = []
    for idx, (depth, frame, K, pose_wc) in enumerate(zip(depth_maps, frames, intrinsics, poses_wc)):
        depth = np.asarray(depth, dtype=np.float32)
        mask = np.isfinite(depth) & (depth > 0)
        valid = int(mask.sum())
        total = int(depth.size)
        ratio_invalid = 1.0 - (valid / total if total else 0.0)
        reproj_error = float("nan")
        reproj_med = float("nan")
        if valid > 0:
            flat_idx = np.flatnonzero(mask)
            take = min(sample_count, flat_idx.size)
            chosen = rng.choice(flat_idx, size=take, replace=False)
            h, w = depth.shape
            v = chosen // w
            u = chosen % w
            z = depth[v, u]
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])
            x = (u.astype(np.float32) - cx) / fx * z
            y = (v.astype(np.float32) - cy) / fy * z
            pts_c = np.stack([x, y, z], axis=1)
            homog = np.concatenate([pts_c, np.ones((pts_c.shape[0], 1), dtype=np.float32)], axis=1)
            pts_w = (pose_wc @ homog.T).T[:, :3]
            pose_cw = np.linalg.inv(pose_wc)
            homog_w = np.concatenate([pts_w, np.ones((pts_w.shape[0], 1), dtype=np.float32)], axis=1)
            pts_cam = (pose_cw @ homog_w.T).T[:, :3]
            positive = pts_cam[:, 2] > 1e-6
            if np.any(positive):
                pts_cam = pts_cam[positive]
                u_valid = u[positive].astype(np.float32)
                v_valid = v[positive].astype(np.float32)
                u_proj = fx * (pts_cam[:, 0] / pts_cam[:, 2]) + cx
                v_proj = fy * (pts_cam[:, 1] / pts_cam[:, 2]) + cy
                diff = np.sqrt((u_proj - u_valid) ** 2 + (v_proj - v_valid) ** 2)
                if diff.size > 0:
                    reproj_error = float(np.mean(diff))
                    reproj_med = float(np.median(diff))
        per_view_logs.append(
            "view %s: valid %.2f%%, reproj_px mean=%.4f median=%.4f"
            % (
                frame.camera_id,
                (valid / total * 100.0) if total else 0.0,
                reproj_error,
                reproj_med,
            )
        )
        packed.append(
            {
                "camera": frame.camera_id,
                "depth": depth,
                "mask": mask,
                "K": K,
                "pose_wc": pose_wc,
                "pose_cw": np.linalg.inv(pose_wc),
            }
        )

    for line in per_view_logs:
        LOGGER.info("Debug geo (single view): %s", line)

    cross_logs: List[str] = []
    for i in range(len(packed)):
        for j in range(len(packed)):
            if i == j:
                continue
            src = packed[i]
            dst = packed[j]
            valid_idx = np.flatnonzero(src["mask"])
            if valid_idx.size == 0:
                continue
            take = min(sample_count // 2, valid_idx.size)
            chosen = rng.choice(valid_idx, size=take, replace=False)
            h, w = src["depth"].shape
            v_src = chosen // w
            u_src = chosen % w
            z_src = src["depth"][v_src, u_src]
            fx_s, fy_s = float(src["K"][0, 0]), float(src["K"][1, 1])
            cx_s, cy_s = float(src["K"][0, 2]), float(src["K"][1, 2])
            x_s = (u_src.astype(np.float32) - cx_s) / fx_s * z_src
            y_s = (v_src.astype(np.float32) - cy_s) / fy_s * z_src
            pts_cam_src = np.stack([x_s, y_s, z_src], axis=1)
            homog_src = np.concatenate([pts_cam_src, np.ones((pts_cam_src.shape[0], 1), dtype=np.float32)], axis=1)
            pts_w = (src["pose_wc"] @ homog_src.T).T[:, :3]
            homog_w = np.concatenate([pts_w, np.ones((pts_w.shape[0], 1), dtype=np.float32)], axis=1)
            pts_cam_dst = (dst["pose_cw"] @ homog_w.T).T[:, :3]
            positive = pts_cam_dst[:, 2] > 1e-6
            if not np.any(positive):
                continue
            pts_cam_dst = pts_cam_dst[positive]
            pts_w = pts_w[positive]
            u_src = u_src[positive].astype(np.float32)
            v_src = v_src[positive].astype(np.float32)
            fx_d, fy_d = float(dst["K"][0, 0]), float(dst["K"][1, 1])
            cx_d, cy_d = float(dst["K"][0, 2]), float(dst["K"][1, 2])
            u_proj = fx_d * (pts_cam_dst[:, 0] / pts_cam_dst[:, 2]) + cx_d
            v_proj = fy_d * (pts_cam_dst[:, 1] / pts_cam_dst[:, 2]) + cy_d
            inside = (u_proj >= 0) & (u_proj <= dst["depth"].shape[1] - 1) & (v_proj >= 0) & (
                v_proj <= dst["depth"].shape[0] - 1
            )
            if not np.any(inside):
                continue
            u_proj = u_proj[inside]
            v_proj = v_proj[inside]
            pts_cam_dst = pts_cam_dst[inside]
            pts_w = pts_w[inside]
            u_src = u_src[inside]
            v_src = v_src[inside]
            depth_interp = _bilinear_sample(dst["depth"], u_proj, v_proj)
            finite = np.isfinite(depth_interp)
            if not np.any(finite):
                continue
            depth_interp = depth_interp[finite]
            u_proj = u_proj[finite]
            v_proj = v_proj[finite]
            pts_cam_dst = pts_cam_dst[finite]
            pts_w = pts_w[finite]
            u_src = u_src[finite]
            v_src = v_src[finite]
            x_obs = (u_proj - cx_d) / fx_d * depth_interp
            y_obs = (v_proj - cy_d) / fy_d * depth_interp
            pts_cam_obs = np.stack([x_obs, y_obs, depth_interp], axis=1)
            homog_obs = np.concatenate([pts_cam_obs, np.ones((pts_cam_obs.shape[0], 1), dtype=np.float32)], axis=1)
            pts_w_obs = (dst["pose_wc"] @ homog_obs.T).T[:, :3]
            homog_back = np.concatenate([pts_w_obs, np.ones((pts_w_obs.shape[0], 1), dtype=np.float32)], axis=1)
            pts_cam_back = (src["pose_cw"] @ homog_back.T).T[:, :3]
            positive_back = pts_cam_back[:, 2] > 1e-6
            if not np.any(positive_back):
                continue
            pts_cam_back = pts_cam_back[positive_back]
            pts_w_obs = pts_w_obs[positive_back]
            pts_w = pts_w[positive_back]
            u_src = u_src[positive_back]
            v_src = v_src[positive_back]
            u_back = fx_s * (pts_cam_back[:, 0] / pts_cam_back[:, 2]) + cx_s
            v_back = fy_s * (pts_cam_back[:, 1] / pts_cam_back[:, 2]) + cy_s
            err_px = np.sqrt((u_back - u_src) ** 2 + (v_back - v_src) ** 2)
            err_world = np.linalg.norm(pts_w_obs - pts_w, axis=1)
            if err_px.size == 0:
                continue
            cross_logs.append(
                "view %s -> %s: reproj_px mean=%.4f median=%.4f | world mm mean=%.2f"
                % (
                    src["camera"],
                    dst["camera"],
                    float(np.mean(err_px)),
                    float(np.median(err_px)),
                    float(np.mean(err_world) * 1000.0),
                )
            )
    for line in cross_logs:
        LOGGER.info("Debug geo (cross view): %s", line)


class DeepViewFrameProvider(io_utils.FrameProvider):
    """
    Frame provider that wraps ``DeepViewDataset`` objects and exposes the
    legacy FrameProvider interface expected by the pipeline.
    """

    def __init__(
        self,
        dataset: io_deepview.DeepViewDataset,
        camera_ids: Sequence[str],
        *,
        start_frame: int,
        frame_stride: int,
        max_batches: Optional[int],
    ) -> None:
        self.dataset = dataset
        self.camera_ids = list(camera_ids)
        self.start_frame = max(0, int(start_frame))
        self.frame_stride = max(1, int(frame_stride))
        self.max_batches = max_batches
        self._next_frame = self.start_frame + self.frame_stride

    def bootstrap(self, num_cams: int) -> List[io_utils.FrameData]:
        frames: List[io_utils.FrameData] = []
        count = min(num_cams, len(self.camera_ids))
        if count == 0:
            raise RuntimeError("DeepView dataset produced no cameras for bootstrap.")
        for cam_id in self.camera_ids[:count]:
            image = self.dataset.get_frame(cam_id, self.start_frame)
            frames.append(
                io_utils.FrameData(
                    camera_id=cam_id,
                    frame_id=self.start_frame,
                    image=image,
                    path=None,
                    timestamp=None,
                )
            )
        return frames

    def iter_batches(self) -> Iterator[io_utils.FrameBatch]:
        batch_idx = 0
        frame_idx = self._next_frame
        while True:
            frames: Dict[str, io_utils.FrameData] = {}
            try:
                for cam_id in self.camera_ids:
                    image = self.dataset.get_frame(cam_id, frame_idx)
                    frames[cam_id] = io_utils.FrameData(
                        camera_id=cam_id,
                        frame_id=frame_idx,
                        image=image,
                        path=None,
                        timestamp=None,
                    )
            except RuntimeError:
                break
            yield io_utils.FrameBatch(index=batch_idx, frames=frames)
            batch_idx += 1
            if self.max_batches is not None and batch_idx >= self.max_batches:
                break
            frame_idx += self.frame_stride
        self._next_frame = frame_idx

    def close(self) -> None:
        self.dataset.close()


def _save_rectified_intrinsics(out_dir: Path, intrinsics: Dict[str, np.ndarray]) -> None:
    target = out_dir / "calibration_rectified"
    target.mkdir(parents=True, exist_ok=True)
    for cam_id, matrix in intrinsics.items():
        payload = {
            "camera": cam_id,
            "matrix": matrix.tolist(),
        }
        path = target / f"{cam_id}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


REFERENCE_WORLD_POINT = np.array([0.04770101, 0.04799868, 1.28055], dtype=np.float32)


def _validate_projection(cam_id: str, params: Dict[str, np.ndarray]) -> None:
    """
    Project the DeepView README reference world-space point and log the pixel
    coordinates.  Expected result is approximately [1377.855, 1017.614].
    """
    world = REFERENCE_WORLD_POINT
    R = params["R"]
    t = params["t"]
    point_c = R @ world + t
    x = float(point_c[0])
    y = float(point_c[1])
    z = float(point_c[2])
    r = float(np.hypot(x, y))
    theta = float(np.arctan2(r, z))
    if r < 1e-8:
        theta_over_r = 1.0
    else:
        theta_over_r = theta / r
    r2 = theta * theta
    k1, k2, k3 = params["distortion"]
    distortion = 1.0 + r2 * (float(k1) + r2 * float(k2))
    distortion += (r2 * r2) * float(k3)
    x_norm = theta_over_r * x * distortion
    y_norm = theta_over_r * y * distortion
    K_raw = params["K_raw"]
    u = float(K_raw[0, 0] * x_norm + K_raw[0, 2])
    v = float(K_raw[1, 1] * y_norm + K_raw[1, 2])
    LOGGER.info("Projection check for %s -> [%.3f, %.3f]", cam_id, u, v)


def _build_generic_provider(
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[io_utils.FrameProvider, List[io_utils.FrameData]]:
    requested_views = _parse_view_list(args.views)
    num_cams = max(1, int(args.n_views))
    if args.source == "images":
        root = args.images_live or args.images
        if root is None:
            raise RuntimeError("--images (or --images_live) required when --source=images")
    else:
        root = None

    if args.source == "images":
        provider = io_utils.ImageSequenceProvider(
            Path(root),
            num_cams=num_cams,
            requested_views=requested_views,
            random_views=getattr(args, "random_views", 0),
            frame_step=args.frame_stride,
            max_batches=args.max_batches,
            rng=rng,
        )
    elif args.source == "videos":
        if args.videos is None:
            raise RuntimeError("--videos is required when --source=videos")
        videos = io_utils.discover_video_files(Path(args.videos))
        provider = io_utils.VideoStreamProvider(
            videos,
            num_cams=num_cams,
            requested_views=requested_views,
            random_views=getattr(args, "random_views", 0),
            frame_step=args.frame_stride,
            max_batches=args.max_batches,
            rng=rng,
        )
    elif args.source == "webcams":
        if not args.webcams:
            raise RuntimeError("--webcams is required when --source=webcams")
        webcams = io_utils.parse_webcam_spec(args.webcams)
        selected = webcams
        if requested_views:
            missing = [cam for cam in requested_views if cam not in webcams]
            if missing:
                raise RuntimeError(f"Requested webcams not available: {missing}")
            selected = {cam: webcams[cam] for cam in requested_views}
        elif getattr(args, "random_views", 0) > 0:
            if args.random_views > len(webcams):
                raise RuntimeError("Cannot sample more webcams than available.")
            chosen = rng.sample(list(webcams.keys()), args.random_views)
            selected = {cam: webcams[cam] for cam in chosen}
        if num_cams < len(selected):
            chosen = list(selected.keys())[:num_cams]
            selected = {cam: selected[cam] for cam in chosen}
        provider = io_utils.WebcamStreamProvider(
            selected,
            frame_step=args.frame_stride,
            max_batches=args.max_batches,
        )
    else:  # pragma: no cover - parser guards choices
        raise RuntimeError(f"Unsupported source type: {args.source}")

    bootstrap_frames = provider.bootstrap(num_cams)
    if not bootstrap_frames:
        raise RuntimeError("No frames available for VGGT bootstrap.")
    return provider, bootstrap_frames


def _load_initial_map(
    path: Optional[Path],
    fusion_mode: str,
    voxel_size: float,
    trunc_margin: float,
) -> tuple[Optional[fusion.SurfelMap], Optional[fusion.TSDFVolume]]:
    if path is None:
        return None, None
    path = Path(path)
    if path.suffix.lower() == ".ply":
        points = io_utils.load_point_cloud(path)
        return fusion.SurfelMap(positions=points), None
    if path.suffix.lower() == ".npz":
        coords, values = io_utils.load_tsdf(path)
        origin = coords.min(axis=0) if coords.size else np.zeros(3, dtype=np.float32)
        tsdf = fusion.TSDFVolume(
            origin=origin,
            voxel_size=voxel_size,
            trunc_margin=trunc_margin,
        )
        for coord, value in zip(coords, values):
            idx = tuple(np.floor((coord - tsdf.origin) / tsdf.voxel_size).astype(int))
            tsdf.voxels[idx] = (value, 1.0)
        surfels = fusion.SurfelMap(positions=tsdf.to_point_cloud())
        return surfels, tsdf
    raise ValueError(f"Unknown map extension: {path}")


def bootstrap_initial_map(
    bootstrap_frames: Sequence[io_utils.FrameData],
    intrinsics: Dict[str, np.ndarray],
    poses: Dict[tuple[str, int], np.ndarray],
    backend: str,
    vggt_engine: Optional[Path],
    vggt_norm: str,
    fusion_mode: str,
    voxel_size: float,
    trunc_margin: float,
    target_hw: Tuple[int, int],
    hf_checkpoint: str,
    hf_weights: Optional[str],
    hf_device: Optional[str],
    hf_fallback: bool,
    using_dataset_cameras: bool,
    log_depth_stats: bool,
    pose_source: str,
    debug_geometry: bool,
) -> tuple[fusion.SurfelMap, Optional[fusion.TSDFVolume], Optional[Tuple[np.ndarray, np.ndarray]]]:
    LOGGER.info("Bootstrapping initial metric map from VGGT (%s backend).", backend)
    if pose_source not in {"auto", "dataset", "vggt"}:
        raise ValueError(f"Unsupported pose source '{pose_source}'.")
    if not bootstrap_frames:
        raise RuntimeError("Bootstrap requires at least one frame.")
    fallback_shape = bootstrap_frames[0].image.shape[:2]

    hf_outputs: Optional[List[dict]] = None
    trt_result: Optional[vggt_trt.VGGTOutput] = None
    if backend == "huggingface":
        start = time.perf_counter()
        hf_outputs = _run_vggt_hf(
            bootstrap_frames,
            target_hw=target_hw,
            checkpoint=hf_checkpoint,
            weight_override=hf_weights,
            device_override=hf_device,
        )
        duration = (time.perf_counter() - start) * 1000.0
        LOGGER.info("HuggingFace VGGT inference completed in %.1f ms", duration)
        depth_sources = [output["depth"] for output in hf_outputs]
    else:
        if vggt_engine is None:
            raise RuntimeError("--vggt_engine is required when --vggt-backend=trt")
        images = [frame.image for frame in bootstrap_frames]
        vggt = vggt_trt.TRTVGGT(engine=vggt_engine, input_normalization=vggt_norm)
        start = time.perf_counter()
        fallback_norms = tuple(
            norm_name
            for norm_name in ("imagenet", "zero_center", "minus_one_to_one", "tanh")
            if norm_name != vggt_norm
        )
        try:
            trt_result = vggt.run(images, norm=vggt_norm, retry_norms=fallback_norms)
        except RuntimeError as exc:
            raise RuntimeError(f"TensorRT VGGT inference failed: {exc}") from exc
        duration = (time.perf_counter() - start) * 1000.0
        LOGGER.info("TensorRT VGGT inference completed in %.1f ms", duration)
        depth_sources = trt_result.depth_maps or [np.ones(images[0].shape[:2], dtype=np.float32)] * len(images)
        if trt_result.camera_extrinsics is None or trt_result.camera_intrinsics is None:
            if hf_fallback and hf_weights:
                LOGGER.info(
                    "TensorRT engine did not expose camera parameters; running HuggingFace VGGT to recover poses."
                )
                start = time.perf_counter()
                hf_outputs = _run_vggt_hf(
                    bootstrap_frames,
                    target_hw=target_hw,
                    checkpoint=hf_checkpoint,
                    weight_override=hf_weights,
                    device_override=hf_device,
                )
                LOGGER.info(
                    "HuggingFace pose recovery completed in %.1f ms",
                    (time.perf_counter() - start) * 1000.0,
                )
            elif using_dataset_cameras:
                LOGGER.debug("TensorRT engine omitted camera parameters; falling back to dataset-supplied values.")
            else:
                LOGGER.warning(
                    "TensorRT engine omitted camera parameters and no fallback was requested."
                )

    sanitized: List[np.ndarray] = []
    for idx, d in enumerate(depth_sources):
        arr = np.asarray(d, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.ndim != 2:
            LOGGER.warning("Depth map %d has unexpected shape %s; squeezing may be required.", idx, arr.shape)
            arr = np.squeeze(arr)
        finite = np.isfinite(arr)
        pos = finite & (arr > 0)
        mn = float(np.nanmin(arr)) if np.any(finite) else float("nan")
        mx = float(np.nanmax(arr)) if np.any(finite) else float("nan")
        cnt_pos = int(np.count_nonzero(pos))
        total = int(arr.size)
        LOGGER.debug(
            "Depth[%d]: shape=%s min=%.4f max=%.4f pos=%d/%d",
            idx,
            tuple(arr.shape),
            mn,
            mx,
            cnt_pos,
            total,
        )
        if cnt_pos == 0:
            LOGGER.warning(
                "Depth[%d] contains no positive finite values (min=%.4f max=%.4f); replacing with ones.",
                idx,
                mn,
                mx,
            )
            arr = np.ones(fallback_shape, dtype=np.float32)
        sanitized.append(arr.astype(np.float32))
    depth_maps = sanitized

    if log_depth_stats:
        for frame, depth in zip(bootstrap_frames, depth_maps):
            finite = depth[np.isfinite(depth)]
            if finite.size == 0:
                LOGGER.info("Depth[%s]: no finite values", frame.camera_id)
            else:
                LOGGER.info(
                    "Depth[%s]: min=%.4f max=%.4f mean=%.4f",
                    frame.camera_id,
                    float(np.min(finite)),
                    float(np.max(finite)),
                    float(np.mean(finite)),
                )

    if hf_outputs is not None:
        for output, depth in zip(hf_outputs, depth_maps):
            output["depth"] = depth

    if backend == "huggingface" and hf_outputs is not None:
        for frame, output in zip(bootstrap_frames, hf_outputs):
            intrinsic_hf = output.get("intrinsic")
            if intrinsic_hf is not None:
                intrinsics[frame.camera_id] = intrinsic_hf.astype(np.float32)
            extr_hf = output.get("extrinsic")
            if extr_hf is not None:
                extr4 = np.eye(4, dtype=np.float32)
                extr4[:3, :3] = extr_hf[:, :3]
                extr4[:3, 3] = extr_hf[:, 3]
                cam_to_world = np.linalg.inv(extr4)
                poses[(frame.camera_id, frame.frame_id)] = cam_to_world.astype(np.float32)
                poses[(frame.camera_id, 0)] = cam_to_world.astype(np.float32)

    elif backend == "trt" and hf_outputs is not None:
        for frame, output in zip(bootstrap_frames, hf_outputs):
            intrinsic_hf = output.get("intrinsic")
            if intrinsic_hf is not None:
                intrinsics[frame.camera_id] = intrinsic_hf.astype(np.float32)
            extr_hf = output.get("extrinsic")
            if extr_hf is not None:
                extr4 = np.eye(4, dtype=np.float32)
                extr4[:3, :3] = extr_hf[:, :3]
                extr4[:3, 3] = extr_hf[:, 3]
                cam_to_world = np.linalg.inv(extr4)
                poses[(frame.camera_id, frame.frame_id)] = cam_to_world.astype(np.float32)
                poses[(frame.camera_id, 0)] = cam_to_world.astype(np.float32)
    elif backend == "trt" and trt_result is not None:
        extr_arr = trt_result.camera_extrinsics
        intr_arr = trt_result.camera_intrinsics
        if pose_source == "vggt" and extr_arr is None:
            raise RuntimeError(
                "TensorRT engine did not expose camera extrinsics but --pose-source=vggt was requested."
            )
        if extr_arr is not None and extr_arr.ndim == 4:
            extr_arr = extr_arr[0]
        if intr_arr is not None and intr_arr.ndim == 4:
            intr_arr = intr_arr[0]
        use_trt_extr = pose_source in {"auto", "vggt"} and extr_arr is not None
        need_intrinsics = {frame.camera_id for frame in bootstrap_frames if frame.camera_id not in intrinsics}
        use_trt_intr = intr_arr is not None and (pose_source in {"auto", "vggt"} or bool(need_intrinsics))
        if not use_trt_extr and pose_source == "vggt":
            raise RuntimeError(
                "TensorRT engine omitted camera extrinsics required by --pose-source=vggt."
            )
        if use_trt_intr or use_trt_extr:
            LOGGER.info(
                "Applying TensorRT camera parameters (extrinsics=%s, intrinsics=%s).",
                "yes" if use_trt_extr else "no",
                "yes" if use_trt_intr else "no",
            )
            for idx, frame in enumerate(bootstrap_frames):
                if use_trt_intr and intr_arr is not None and idx < intr_arr.shape[0]:
                    intrinsics[frame.camera_id] = intr_arr[idx].astype(np.float32)
                if use_trt_extr and extr_arr is not None and idx < extr_arr.shape[0]:
                    extr = extr_arr[idx]
                    extr4 = np.eye(4, dtype=np.float32)
                    extr4[:3, :4] = extr
                    cam_to_world = np.linalg.inv(extr4)
                    poses[(frame.camera_id, frame.frame_id)] = cam_to_world.astype(np.float32)
                    poses[(frame.camera_id, 0)] = cam_to_world.astype(np.float32)
        elif pose_source == "dataset":
            LOGGER.debug("Retaining dataset camera parameters (pose_source=dataset).")

    Ks: List[np.ndarray] = []
    missing_intr = [frame.camera_id for frame in bootstrap_frames if frame.camera_id not in intrinsics]
    if missing_intr:
        raise RuntimeError(
            "Missing intrinsics for cameras {}. Re-export the TensorRT engine with pose outputs, "
            "provide dataset cameras via --use-dataset-cameras, or enable --hf-fallback with --hf-weights.".format(
                ", ".join(missing_intr)
            )
        )
    for frame in bootstrap_frames:
        if frame.camera_id not in intrinsics:
            raise RuntimeError(f"Missing intrinsics for camera {frame.camera_id}")
        Ks.append(intrinsics[frame.camera_id])
    poses_wc = []
    missing_pose = [frame.camera_id for frame in bootstrap_frames if (frame.camera_id, frame.frame_id) not in poses and (frame.camera_id, 0) not in poses]
    if missing_pose:
        raise RuntimeError(
            "Missing poses for cameras {}. Re-export the TensorRT engine with pose outputs, provide "
            "dataset cameras via --use-dataset-cameras, or enable --hf-fallback.".format(
                ", ".join(missing_pose)
            )
        )
    for frame in bootstrap_frames:
        pose = poses.get((frame.camera_id, frame.frame_id))
        if pose is None:
            pose = poses.get((frame.camera_id, 0))
        if pose is None:
            raise RuntimeError(f"Missing pose for bootstrap frame {frame.camera_id}/{frame.frame_id}")
        poses_wc.append(pose)
    if debug_geometry:
        _geometry_debug_report(depth_maps, bootstrap_frames, Ks, poses_wc)
    point_batches: List[np.ndarray] = []
    color_batches: List[np.ndarray] = []
    if hf_outputs is not None and backend in {"huggingface", "trt"}:
        for depth, frame, output in zip(depth_maps, bootstrap_frames, hf_outputs):
            world_points = output.get("point_map_unprojected")
            rgb = output.get("rgb")
            if world_points is None or rgb is None:
                continue
            valid = np.isfinite(depth) & (depth > 0)
            pts = world_points[valid]
            cols = (rgb[valid] * 255.0).clip(0, 255).astype(np.uint8)
            if pts.size > 0:
                point_batches.append(pts.reshape(-1, 3))
                color_batches.append(cols.reshape(-1, 3))
    else:
        for depth, frame, K, pose in zip(depth_maps, bootstrap_frames, Ks, poses_wc):
            pts, cols = _points_and_colors_from_depth(depth, frame.image, K, pose)
            if pts.size > 0:
                point_batches.append(pts)
                color_batches.append(cols)
    bootstrap_cloud: Optional[Tuple[np.ndarray, np.ndarray]] = None
    if point_batches:
        bootstrap_cloud = (
            np.concatenate(point_batches, axis=0),
            np.concatenate(color_batches, axis=0),
        )
    surfels = fusion.SurfelMap.from_depth_batch(tuple(depth_maps), tuple(Ks), tuple(poses_wc))
    tsdf_volume: Optional[fusion.TSDFVolume] = None
    if fusion_mode == "tsdf":
        tsdf_volume = fusion.TSDFVolume(
            origin=surfels.positions.min(axis=0) if surfels.positions.size else np.zeros(3, dtype=np.float32),
            voxel_size=voxel_size,
            trunc_margin=trunc_margin,
        )
        for depth, K, pose in zip(depth_maps, Ks, poses_wc):
            tsdf_volume.integrate(depth, K, pose)
        surfels = fusion.SurfelMap(positions=tsdf_volume.to_point_cloud())
    return surfels, tsdf_volume, bootstrap_cloud


def run_pipeline(args: argparse.Namespace) -> int:
    _normalize_args(args)
    logging.basicConfig(level=getattr(logging, args.log_level))
    if args.pose_source == "dataset" and not args.use_dataset_cameras:
        LOGGER.info("Enabling --use-dataset-cameras because --pose-source=dataset was requested.")
        args.use_dataset_cameras = True
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rectify_width, rectify_height = map(int, args.rectify_size)
    rectify_size = (rectify_width, rectify_height)

    calib_path = Path(args.per_camera_calib) if args.per_camera_calib else None
    scale_cache = io_utils.load_per_camera_calibration(calib_path)
    process_noise = tuple(float(x) for x in args.align_process_noise)
    if args.smooth_scale > 0.0:
        LOGGER.warning("--smooth_scale is deprecated; EKF-based smoothing is used instead.")
    metric_aligner = metric_depth.MetricDepthAligner(
        strategy=args.align_model,
        max_samples=int(args.align_max_samples),
        process_noise=process_noise,
        measurement_floor=float(args.align_measurement_floor),
        tsdf_influence=float(args.align_tsdf_weight),
        rng_seed=int(args.seed),
    )
    for cam_id, (scale, shift) in scale_cache.items():
        metric_aligner.set_state(cam_id, scale, shift)

    intrinsics: Dict[str, np.ndarray] = {}
    poses: Dict[tuple[str, int], np.ndarray] = {}
    resize_targets: Dict[str, Tuple[int, int]] = {}
    raw_params: Dict[str, Dict[str, np.ndarray]] = {}

    rng = random.Random(args.seed)
    if args.dataset == "deepview":
        if args.dataset_root is None or args.scene is None:
            raise RuntimeError("--dataset-root and --scene are required for DeepView datasets.")
        dataset = io_deepview.DeepViewDataset(
            root=args.dataset_root,
            scene=args.scene,
            undistort=args.undistort.lower() != "off",
            rectify_to_size=rectify_size,
            cache_maps=args.cache_rectify_maps,
            seed=args.seed,
            frame_stride=args.frame_stride,
            default_frame=args.frame_index,
        )
        available = dataset.list_cameras()
        requested = _parse_view_list(args.views)
        if requested:
            missing = [cam for cam in requested if cam not in available]
            if missing:
                raise RuntimeError(f"Requested DeepView cameras not found: {missing}")
            selected = requested[:]
        else:
            selected = io_deepview.select_views(
                available,
                args.n_views,
                method=args.view_select,
                seed=args.seed,
            )
        provider: io_utils.FrameProvider = DeepViewFrameProvider(
            dataset,
            selected,
            start_frame=args.frame_index,
            frame_stride=args.frame_stride,
            max_batches=args.max_batches,
        )
        bootstrap_frames = provider.bootstrap(len(selected))
        if not bootstrap_frames:
            raise RuntimeError("No frames available for VGGT bootstrap.")
        for cam_id in selected:
            params = dataset.get_camera_params(cam_id)
            raw_params[cam_id] = params
            if args.use_dataset_cameras:
                intrinsics[cam_id] = params["K"]
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = params["R"].T
                pose[:3, 3] = params["center"]
                poses[(cam_id, 0)] = pose
                poses[(cam_id, args.frame_index)] = pose
        active_cams = [frame.camera_id for frame in bootstrap_frames]
        LOGGER.info("Active DeepView cameras: %s", ", ".join(active_cams))
        if args.use_dataset_cameras:
            _save_rectified_intrinsics(out_dir, intrinsics)
            if args.validate_projection and active_cams:
                first_cam = active_cams[0]
                _validate_projection(first_cam, raw_params[first_cam])
    else:
        if args.intrinsics is None or args.poses is None:
            raise RuntimeError("Generic datasets require --intrinsics and --poses.")
        intrinsics_raw = io_utils.load_intrinsics(Path(args.intrinsics))
        poses = io_utils.load_poses(Path(args.poses))
        provider, bootstrap_frames = _build_generic_provider(args, rng)
        if not bootstrap_frames:
            raise RuntimeError("No frames available for VGGT bootstrap.")
        active_cams = [frame.camera_id for frame in bootstrap_frames]
        LOGGER.info("Active generic cameras: %s", ", ".join(active_cams))
        for frame in bootstrap_frames:
            cam_id = frame.camera_id
            image = frame.image
            h, w = image.shape[:2]
            scale_x = rectify_width / float(w)
            scale_y = rectify_height / float(h)
            if (w, h) != (rectify_width, rectify_height):
                resize_targets[cam_id] = (rectify_width, rectify_height)
                frame.image = _resize_image(image, rectify_width, rectify_height)
            K_raw = intrinsics_raw.get(cam_id)
            if K_raw is None:
                raise RuntimeError(f"Missing intrinsics for camera {cam_id}")
            intrinsics[cam_id] = _scale_intrinsics_local(K_raw, scale_x, scale_y)
        _save_rectified_intrinsics(out_dir, intrinsics)

    surfel_map, tsdf_map = _load_initial_map(
        Path(args.initial_map) if args.initial_map else None,
        args.fusion,
        args.voxel_size,
        args.trunc_margin,
    )

    if surfel_map is None and tsdf_map is None:
        if not args.test_mode:
            raise RuntimeError("--initial_map is required unless --test-mode=1")
        if args.vggt_backend == "trt" and args.vggt_engine is None:
            raise RuntimeError("--vggt_engine must be provided when --vggt-backend=trt")
        LOGGER.info(
            "VGGT input tensor shape: (1, %d, 3, %d, %d)",
            len(bootstrap_frames),
            rectify_height,
            rectify_width,
        )
        surfel_map, tsdf_map, bootstrap_cloud = bootstrap_initial_map(
            bootstrap_frames,
            intrinsics,
            poses,
            args.vggt_backend,
            Path(args.vggt_engine) if args.vggt_engine else None,
            args.vggt_norm,
            args.fusion,
            args.voxel_size,
            args.trunc_margin,
            (rectify_height, rectify_width),
            args.hf_checkpoint,
            args.hf_weights,
            args.hf_device,
            bool(args.hf_fallback),
            bool(args.use_dataset_cameras),
            bool(args.log_depth_stats),
            args.pose_source,
            bool(args.debug_geo),
        )
        io_utils.save_point_cloud(out_dir / "bootstrap_map.ply", surfel_map.as_point_cloud())
        if bootstrap_cloud is not None:
            pts_boot, cols_boot = bootstrap_cloud
            LOGGER.info("Saving VGGT stage point cloud with %d points.", pts_boot.shape[0])
            io_utils.save_point_cloud(out_dir / "vggt_reconstruction.ply", pts_boot, cols_boot)
        else:
            LOGGER.warning("VGGT bootstrap produced no valid point cloud for color export.")

    depth_workers = args.depth_workers or len(active_cams)
    metrics: List[dict] = []
    depth_stage_saved = False
    explicit_depth_engine = Path(args.depth_engine) if getattr(args, "depth_engine", None) else None
    try:
        with depth_anything.DepthAnythingPool(
            Path(args.depth_engine_dir),
            precision=args.trt_precision,
            num_workers=max(1, int(depth_workers)),
            explicit_engine=explicit_depth_engine,
        ) as depth_pool:
            LOGGER.info("Depth Anything engine: %s", depth_pool.engine_path)
            for batch in provider.iter_batches():
                batch_start = time.perf_counter()
                if not batch.frames:
                    continue
                updated_this_batch = False
                batch_points: List[np.ndarray] = []
                batch_colors: List[np.ndarray] = []
                for cam_id, frame in batch.frames.items():
                    if cam_id in resize_targets:
                        width, height = resize_targets[cam_id]
                        frame.image = _resize_image(frame.image, width, height)
                images = {cam_id: frame.image for cam_id, frame in batch.frames.items()}
                depth_start = time.perf_counter()
                rel_depths = depth_pool.infer_batch(images)
                depth_ms = (time.perf_counter() - depth_start) * 1000.0

                for cam_id, frame in batch.frames.items():
                    K = intrinsics.get(cam_id)
                    pose = poses.get((cam_id, frame.frame_id))
                    if pose is None:
                        pose = poses.get((cam_id, 0))
                    if K is None or pose is None:
                        LOGGER.warning("Missing calibration for %s frame %d", cam_id, frame.frame_id)
                        continue
                    rel_depth = rel_depths.get(cam_id)
                    if rel_depth is None:
                        LOGGER.warning("Depth Anything produced no output for camera %s", cam_id)
                        continue

                    if args.fusion == "tsdf" and tsdf_map is not None:
                        expected = raycast.expected_depth_from_surfel(
                            tsdf_map.to_point_cloud(),
                            pose,
                            K,
                            frame.image.shape[:2],
                        )
                    else:
                        expected = raycast.expected_depth_from_surfel(
                            surfel_map.positions if surfel_map is not None else np.empty((0, 3), dtype=np.float32),
                            pose,
                            K,
                            frame.image.shape[:2],
                        )

                    tsdf_sampler = None
                    if args.fusion == "tsdf" and tsdf_map is not None:
                        tsdf_sampler = tsdf_map.sample
                    try:
                        alignment = metric_aligner.align(
                            cam_id,
                            rel_depth,
                            expected,
                            K,
                            pose,
                            mask=np.isfinite(expected),
                            tsdf_sampler=tsdf_sampler,
                        )
                    except ValueError as exc:
                        LOGGER.warning(
                            "Alignment skipped for batch %d camera %s: %s",
                            batch.index,
                            cam_id,
                            exc,
                        )
                        continue
                    metric_depth_map = alignment.apply(rel_depth)

                    if args.fusion == "tsdf":
                        if tsdf_map is None:
                            tsdf_map = fusion.TSDFVolume(
                                origin=np.zeros(3, dtype=np.float32),
                                voxel_size=args.voxel_size,
                                trunc_margin=args.trunc_margin,
                            )
                        tsdf_map.integrate(metric_depth_map, K, pose)
                        surfel_map = fusion.SurfelMap(positions=tsdf_map.to_point_cloud())
                        updated_this_batch = True
                    else:
                        surfel_map.fuse(metric_depth_map, K, pose)
                        updated_this_batch = True
                    pts_stage, cols_stage = _points_and_colors_from_depth(metric_depth_map, frame.image, K, pose)
                    if pts_stage.size > 0:
                        batch_points.append(pts_stage)
                        batch_colors.append(cols_stage)

                    if args.save_per_frame_depth:
                        depth_dir = out_dir / "depth_metric"
                        depth_dir.mkdir(parents=True, exist_ok=True)
                        io_utils.save_depth(
                            depth_dir / f"camera_{cam_id}_batch_{batch.index:05d}.npz",
                            metric_depth_map,
                        )

                    metrics.append(
                        {
                            "batch": batch.index,
                            "camera": cam_id,
                            "frame": int(frame.frame_id),
                            "model": alignment.model.value,
                            "scale": alignment.scale,
                            "shift": alignment.shift,
                            "residual": alignment.residual,
                            "depth_ms": depth_ms,
                            "total_ms": (time.perf_counter() - batch_start) * 1000.0,
                        }
                    )
                if updated_this_batch and surfel_map is not None and not depth_stage_saved:
                    if batch_points:
                        pts_stage = np.concatenate(batch_points, axis=0)
                        cols_stage = np.concatenate(batch_colors, axis=0)
                        LOGGER.info("Saving Depth Anything stage point cloud with %d points.", pts_stage.shape[0])
                        io_utils.save_point_cloud(
                            out_dir / "depth_anything_reconstruction.ply",
                            pts_stage,
                            cols_stage,
                        )
                        depth_stage_saved = True
    finally:
        provider.close()

    if surfel_map is None:
        raise RuntimeError("Fusion produced no map output.")

    io_utils.save_point_cloud(out_dir / "map_updated.ply", surfel_map.as_point_cloud())
    if tsdf_map is not None:
        coords, values = tsdf_map.export()
        io_utils.save_tsdf(out_dir / "tsdf_volume.npz", coords, values)
    io_utils.save_metrics(out_dir / "metrics.json", metrics)
    persisted = {
        cam: state
        for cam in intrinsics.keys()
        for state in [metric_aligner.get_state(cam)]
        if state is not None
    }
    io_utils.save_per_camera_calibration(calib_path, persisted)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_pipeline(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
