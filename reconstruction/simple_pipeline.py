"""
Simplified point-cloud pipeline using VGGT + Depth Anything.

The goal is to make the demo_colmap workflow accessible from the command line
with minimal dependencies:

1. VGGT produces metric depth, camera intrinsics/extrinsics and a point map
   obtained via depth unprojection.
2. Depth Anything v2 (ONNX) predicts relative depth for the same frames.  We
   recover metric depth by fitting a scale/shift against VGGT's prediction and
   unproject the rectified map.

Both stages work on image sequences or sampled frames from a video file.  The
pipeline writes out two PLY files (`vggt_point_cloud.ply` and
`depth_anything_rectified_point_cloud.ply`) plus metadata describing camera
parameters and fitted scale factors.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import statistics as stats

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

try:  # pragma: no cover - optional dependency validated at runtime
    import cv2
except Exception as exc:  # pragma: no cover
    raise RuntimeError("OpenCV is required for video/image IO. Install with `pip install opencv-python`.") from exc

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from .capture.multi_camera_loader import MultiCameraImageLoader
from .gaussian.gaussian_field import GaussianFieldBuilder
from .optimized.live_state import OptimizedLiveView
from .pcd.depth_anything import DepthAnythingPool
from .pcd.gpu_ops import unproject_depth_map_to_point_map as gpu_unproject_depth_map
from .viz.o3d_stream import Open3DStreamer


PreprocessedTensor = torch.Tensor  # alias for readability

LOGGER = logging.getLogger("reconstruction.simple_pipeline")


@dataclass(slots=True)
class FrameInfo:
    index: int
    camera: str
    camera_index: int


@dataclass(slots=True)
class FrameChunk:
    info: List[FrameInfo]
    tensor: PreprocessedTensor  # (N, 3, H, W) in [0, 1]


def _collect_images(folder: Path) -> List[Path]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    paths: List[Path] = []
    for ext in exts:
        paths.extend(sorted(folder.glob(f"*{ext}")))
    return paths


def _load_images_from_directory(path: Path, stride: int, max_frames: Optional[int]) -> Iterator[Tuple[FrameInfo, np.ndarray]]:
    images = _collect_images(path)
    if not images:
        raise RuntimeError(f"No image files found under {path}.")
    selected = images[:: max(1, stride)]
    if max_frames:
        selected = selected[: max_frames]

    camera_id = path.name
    global_index = 0
    for local_idx, img_path in enumerate(selected):
        frame_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        info = FrameInfo(index=global_index, camera=camera_id, camera_index=local_idx)
        global_index += 1
        yield info, frame_bgr


def _load_frames_from_videos(
    paths: Sequence[Path],
    stride: int,
    max_frames: Optional[int],
) -> Iterator[Tuple[FrameInfo, np.ndarray]]:
    global_index = 0
    for video_path in paths:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        try:
            frame_idx = 0
            emitted = 0
            camera_id = video_path.stem
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx % stride == 0:
                    info = FrameInfo(index=global_index, camera=camera_id, camera_index=emitted)
                    yield info, frame
                    global_index += 1
                    emitted += 1
                    if max_frames and emitted >= max_frames:
                        break
                frame_idx += 1
        finally:
            cap.release()


def _load_multicam_from_images(
    root: Path,
    stride: int,
    max_frames: Optional[int],
    loops: int,
) -> Iterator[Tuple[FrameInfo, np.ndarray]]:
    loader = MultiCameraImageLoader(root, stride=stride, max_steps=max_frames)
    LOGGER.info(
        "Multi-camera image sequence detected: %s | cameras=%d | steps=%d",
        root,
        len(loader.camera_names),
        loader.sequence_length,
    )
    global_index = 0
    for frame in loader.iter_frames(loops=loops):
        info = FrameInfo(index=global_index, camera=frame.camera, camera_index=frame.index)
        global_index += 1
        yield info, frame.image


def _batch_iterator(
    frame_iter: Iterator[Tuple[FrameInfo, np.ndarray]],
    batch_size: int,
) -> Iterator[FrameChunk]:
    info_batch: List[FrameInfo] = []
    frames: List[np.ndarray] = []
    for info, frame in frame_iter:
        info_batch.append(info)
        frames.append(frame)
        if len(frames) == batch_size:
            yield FrameChunk(info=info_batch.copy(), tensor=_preprocess_frames(frames))
            info_batch.clear()
            frames.clear()
    if frames:
        yield FrameChunk(info=info_batch, tensor=_preprocess_frames(frames))


def _parse_video_paths(path_spec: str) -> List[Path]:
    paths: List[Path] = []
    for raw in path_spec.split(","):
        cleaned = " ".join(raw.split())
        if cleaned:
            paths.append(Path(cleaned))
    return paths


def _preprocess_frames(frames_bgr: Sequence[np.ndarray], mode: str = "crop") -> PreprocessedTensor:
    if mode not in {"crop", "pad"}:
        raise ValueError("mode must be 'crop' or 'pad'.")

    target_size = 518
    to_tensor = TF.to_tensor
    tensors: List[torch.Tensor] = []

    # Track shapes to pad consistently when mixing aspect ratios
    max_height = 0
    max_width = 0
    per_image_tensors: List[torch.Tensor] = []

    for frame_bgr in frames_bgr:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        width, height = img.size

        if mode == "pad":
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14
        else:  # crop
            new_width = target_size
            new_height = round(height * (new_width / width) / 14) * 14

        img_resized = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        tensor = to_tensor(img_resized)  # (3, H, W) in [0, 1]

        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            tensor = tensor[:, start_y : start_y + target_size, :]

        if mode == "pad":
            pad_h = target_size - tensor.shape[1]
            pad_w = target_size - tensor.shape[2]
            if pad_h > 0 or pad_w > 0:
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                tensor = torch.nn.functional.pad(
                    tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        per_image_tensors.append(tensor)
        max_height = max(max_height, tensor.shape[1])
        max_width = max(max_width, tensor.shape[2])

    for tensor in per_image_tensors:
        pad_h = max_height - tensor.shape[1]
        pad_w = max_width - tensor.shape[2]
        if pad_h > 0 or pad_w > 0:
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            tensor = torch.nn.functional.pad(
                tensor, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
            )
        tensors.append(tensor)

    return torch.stack(tensors)


class VGGTPredictor:
    def __init__(self, weights_path: Path, device: Optional[str] = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = VGGT()
        state_dict = torch.load(weights_path, map_location="cpu")
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        if self.device.type == "cuda":
            major, _minor = torch.cuda.get_device_capability(self.device)
            self.autocast_dtype = torch.bfloat16 if major >= 8 else torch.float16
        else:
            self.autocast_dtype = None

    def __call__(self, images: PreprocessedTensor, *, to_cpu: bool = True) -> Dict[str, torch.Tensor]:
        images = images.to(self.device, non_blocking=True)
        with torch.no_grad():
            if self.device.type == "cuda" and self.autocast_dtype is not None:
                with torch.cuda.amp.autocast(dtype=self.autocast_dtype):
                    preds = self.model(images)
            else:
                preds = self.model(images)
        if to_cpu:
            return {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in preds.items()}
        return {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in preds.items()}


def _tensor_to_rgb_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    arr = image_tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (arr * 255.0).astype(np.uint8)


def _tensor_to_bgr_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    rgb = _tensor_to_rgb_uint8(image_tensor)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _flatten_points(
    point_map: np.ndarray | torch.Tensor,
    rgb_image: np.ndarray | torch.Tensor,
    mask: Optional[np.ndarray | torch.Tensor] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(point_map, torch.Tensor):
        pts_tensor = point_map
        if mask is None:
            mask_tensor = torch.ones(point_map.shape[:3], dtype=torch.bool, device=point_map.device)
        else:
            mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=point_map.device)
        if isinstance(rgb_image, torch.Tensor):
            rgb_tensor = rgb_image
        else:
            rgb_tensor = torch.as_tensor(rgb_image, device=point_map.device)
        if rgb_tensor.dim() == 3 and rgb_tensor.shape[0] == 3:
            rgb_tensor = rgb_tensor.permute(1, 2, 0)
        pts = pts_tensor[mask_tensor]
        cols = rgb_tensor[mask_tensor]
        return (
            pts.detach().cpu().numpy().astype(np.float32),
            cols.detach().cpu().numpy().astype(np.uint8),
        )

    if mask is None:
        mask = np.ones(point_map.shape[:3], dtype=bool)
    pts = point_map[mask]
    colors = rgb_image[mask]
    return pts.astype(np.float32), colors.astype(np.uint8)


def _update_viz_cache(
    cache: Optional[Tuple[np.ndarray, np.ndarray]],
    new_points: np.ndarray,
    new_colors: np.ndarray,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if cache is None:
        combined_pts = new_points
        combined_cols = new_colors
    else:
        combined_pts = np.concatenate([cache[0], new_points], axis=0)
        combined_cols = np.concatenate([cache[1], new_colors], axis=0)
    if combined_pts.shape[0] > max_points:
        select = np.random.choice(combined_pts.shape[0], max_points, replace=False)
        combined_pts = combined_pts[select]
        combined_cols = combined_cols[select]
    return combined_pts, combined_cols


def _fit_scale_shift(
    reference,
    target,
    mask,
    confidence=None,
    max_iters: int = 3,
) -> Tuple[float, float]:
    """
    Robust weighted scale/shift alignment between reference (VGGT metric depth)
    and target (Depth Anything relative depth).
    """
    if not isinstance(reference, torch.Tensor):
        ref_tensor = torch.as_tensor(reference, dtype=torch.float32)
        tgt_tensor = torch.as_tensor(target, dtype=torch.float32)
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool)
        conf_tensor = (
            torch.as_tensor(confidence, dtype=torch.float32) if confidence is not None else None
        )
        scale, shift = _fit_scale_shift(ref_tensor, tgt_tensor, mask_tensor, conf_tensor, max_iters)
        return scale, shift

    ref_vals = reference[mask]
    tgt_vals = target[mask]
    if confidence is not None:
        weight_vals = confidence[mask].to(reference.dtype)
    else:
        weight_vals = torch.ones_like(ref_vals)

    finite_mask = torch.isfinite(ref_vals) & torch.isfinite(tgt_vals) & torch.isfinite(weight_vals)
    ref_vals = ref_vals[finite_mask]
    tgt_vals = tgt_vals[finite_mask]
    weight_vals = weight_vals[finite_mask].clamp(min=1e-6)

    if ref_vals.numel() < 10 or tgt_vals.numel() < 10:
        raise RuntimeError("Not enough overlapping valid pixels to fit scale/shift.")

    for iteration in range(max_iters):
        weight_sum = torch.sum(weight_vals)
        if weight_sum <= 0:
            raise RuntimeError("Depth Anything scale fit ill-conditioned (weights sum to zero).")
        tgt_mean = torch.sum(weight_vals * tgt_vals) / weight_sum
        ref_mean = torch.sum(weight_vals * ref_vals) / weight_sum
        var = torch.sum(weight_vals * (tgt_vals - tgt_mean) ** 2)
        if var.abs() < 1e-8:
            raise RuntimeError("Depth Anything scale fit ill-conditioned (variance too small).")
        cov = torch.sum(weight_vals * (tgt_vals - tgt_mean) * (ref_vals - ref_mean))
        scale = cov / var
        shift = ref_mean - scale * tgt_mean

        residuals = ref_vals - (scale * tgt_vals + shift)
        if iteration == max_iters - 1:
            break
        mad = torch.median(residuals.abs())
        if mad <= 1e-6:
            break
        threshold = 3.0 * 1.4826 * mad
        inliers = residuals.abs() <= threshold
        if inliers.sum() < 10:
            break
        ref_vals = ref_vals[inliers]
        tgt_vals = tgt_vals[inliers]
        weight_vals = weight_vals[inliers]

    return float(scale.item()), float(shift.item())


def _prepare_metadata() -> Dict[str, list]:
    return {
        "frame_indices": [],
        "frame_cameras": [],
        "frame_camera_indices": [],
        "extrinsics": [],
        "intrinsics": [],
        "depth_anything_scale": [],
        "depth_backend": None,
        "depth_workers": None,
        "depth_attempted": 0,
        "depth_success": 0,
        "vggt_point_count": 0,
        "depth_point_count": 0,
        "runtime_seconds": 0.0,
        "vggt_device": None,
        "depth_reference_frame": None,
        "outputs": [],
        "vggt_depth_min": [],
        "vggt_depth_max": [],
        "depth_anything_raw_min": [],
        "depth_anything_raw_max": [],
        "depth_anything_rect_min": [],
        "depth_anything_rect_max": [],
        "depth_anything_mae": [],
        "depth_anything_rmse": [],
        "chunk_metrics": [],
        "aggregate_metrics": {},
        "first_depth_alignment": None,
        "gaussian_config": None,
        "gaussian_count": 0,
        "gaussian_path": None,
    }


def run_pipeline(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_path = Path(args.vggt_weights).expanduser().resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"VGGT weights not found: {weights_path}")

    predictor = VGGTPredictor(weights_path, device=args.device)
    LOGGER.info(
        "Starting simple pipeline | input=%s (%s) stride=%d batch=%d loops=%d",
        args.path,
        args.input_type,
        args.stride,
        args.batch_size,
        args.image_loops,
    )
    LOGGER.info("VGGT running on device: %s", predictor.device)

    metadata = _prepare_metadata()
    metadata["vggt_device"] = str(predictor.device)
    metadata["live_viz"] = {
        "mode": args.live_viz,
        "source": args.live_viz_source,
        "max_points": int(args.live_viz_max_points),
        "interval": int(args.live_viz_interval),
        "point_size": int(args.live_viz_point_size),
    }
    gaussian_builder: Optional[GaussianFieldBuilder] = None
    gaussian_mode = args.gaussian_init
    if gaussian_mode != "none":
        gaussian_builder = GaussianFieldBuilder(
            voxel_size=float(args.gaussian_voxel_size),
            min_weight=float(args.gaussian_min_points),
            base_variance=float(args.gaussian_base_variance),
            max_gaussians=int(args.gaussian_max_count) if args.gaussian_max_count > 0 else None,
        )
        metadata["gaussian_config"] = {
            "mode": gaussian_mode,
            "voxel_size": float(args.gaussian_voxel_size),
            "min_weight": float(args.gaussian_min_points),
            "base_variance": float(args.gaussian_base_variance),
            "max_count": int(args.gaussian_max_count),
        }
    else:
        metadata["gaussian_config"] = {"mode": "none"}

    optimized_view: Optional[OptimizedLiveView] = None
    if args.optimized_live_view:
        optimized_view = OptimizedLiveView(
            depth_threshold=float(args.optimized_depth_threshold),
            color_threshold=float(args.optimized_color_threshold),
            confidence_threshold=float(args.optimized_confidence_threshold),
            ema_decay=float(args.optimized_ema),
            max_points_per_frame=int(args.live_viz_max_points),
        )
        LOGGER.info(
            "Optimized live view enabled | depth_thresh=%.3f | color_thresh=%.3f | conf_thresh=%.2f",
            args.optimized_depth_threshold,
            args.optimized_color_threshold,
            args.optimized_confidence_threshold,
        )

    total_start = time.perf_counter()
    save_ply = not args.no_save_ply

    depth_pool: Optional[DepthAnythingPool] = None
    depth_workers_cfg = args.depth_workers
    if args.depth_anything != "off":
        depth_model_path = Path(args.depth_anything_engine).expanduser().resolve()
        if depth_workers_cfg <= 0:
            depth_workers_cfg = DepthAnythingPool.suggest_workers(args.depth_backend)
        LOGGER.info(
            "Depth Anything backend request=%s | workers=%d",
            args.depth_backend,
            depth_workers_cfg,
        )
        if depth_model_path.is_file():
            depth_pool = DepthAnythingPool(
                depth_model_path.parent,
                precision="auto",
                explicit_engine=depth_model_path,
                backend=args.depth_backend,
                num_workers=depth_workers_cfg,
                providers=None,
            )
        else:
            depth_pool = DepthAnythingPool(
                depth_model_path,
                precision="auto",
                explicit_engine=None,
                backend=args.depth_backend,
                num_workers=depth_workers_cfg,
                providers=None,
            )
        metadata["depth_backend"] = depth_pool.backend
        metadata["depth_workers"] = depth_pool.num_workers
    else:
        LOGGER.info("Depth Anything stage disabled (--depth-anything off).")

    if args.input_type == "images":
        cached_frames = list(_load_images_from_directory(Path(args.path), args.stride, args.max_frames))
        if not cached_frames:
            raise RuntimeError(f"No images found under {args.path}")
        LOGGER.info("Image sequence detected: %s | frames=%d", args.path, len(cached_frames))
        if args.image_loops <= 1:
            frame_iter = iter(cached_frames)
        else:
            repeated: List[Tuple[FrameInfo, np.ndarray]] = []
            index_counter = 0
            for loop_idx in range(args.image_loops):
                for info, frame in cached_frames:
                    new_info = FrameInfo(
                        index=index_counter,
                        camera=f"{info.camera}_loop{loop_idx}",
                        camera_index=info.camera_index,
                    )
                    repeated.append((new_info, frame.copy()))
                    index_counter += 1
            LOGGER.info(
                "Image loops enabled (%d×) → total virtual frames=%d",
                args.image_loops,
                len(repeated),
            )
            frame_iter = iter(repeated)
    elif args.input_type == "multicam":
        frame_iter = _load_multicam_from_images(
            Path(args.path),
            stride=args.stride,
            max_frames=args.max_frames,
            loops=args.image_loops,
        )
    else:
        video_paths = _parse_video_paths(args.path)
        if not video_paths:
            raise RuntimeError("No video paths provided.")
        frame_iter = _load_frames_from_videos(video_paths, args.stride, args.max_frames)
        LOGGER.info("Video inputs detected: %s", ", ".join(str(p) for p in video_paths))

    frames_processed = 0
    vggt_point_total = 0
    depth_point_total = 0
    depth_attempted = 0
    depth_success = 0
    chunk_index = 0
    first_frame_saved = False
    first_frame_idx: Optional[int] = None

    aggregated_vggt_pts: List[np.ndarray] = []
    aggregated_vggt_cols: List[np.ndarray] = []
    aggregated_depth_pts: List[np.ndarray] = []
    aggregated_depth_cols: List[np.ndarray] = []
    chunk_metrics_list: List[Dict[str, object]] = metadata["chunk_metrics"]  # type: ignore[assignment]
    viz_streamer: Optional[Open3DStreamer] = None
    viz_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None
    viz_interval = max(1, int(args.live_viz_interval))
    if args.live_viz != "none":
        try:
            import open3d  # type: ignore  # noqa: F401
        except Exception as exc:
            LOGGER.error("Open3D not available for live visualization: %s", exc)
        else:
            try:
                viz_streamer = Open3DStreamer(
                    title=f"Live Reconstruction ({args.live_viz_source})",
                    max_points=int(args.live_viz_max_points),
                    point_size=int(args.live_viz_point_size),
                )
                LOGGER.info(
                    "Live visualization enabled (%s) | source=%s | max_points=%d | interval=%d",
                    args.live_viz,
                    args.live_viz_source,
                    args.live_viz_max_points,
                    viz_interval,
                )
            except Exception as exc:
                LOGGER.error("Failed to initialize live visualization: %s", exc)
                viz_streamer = None

    optimized_chunk_log: List[Dict[str, float]] = []

    for chunk in _batch_iterator(frame_iter, args.batch_size):
        chunk_index += 1
        chunk_frames = len(chunk.info)
        frames_processed += chunk_frames
        chunk_reuse_ratios: List[float] = []
        chunk_dynamic_pixels = 0
        chunk_static_pixels = 0

        vggt_start = time.perf_counter()
        preds = predictor(chunk.tensor, to_cpu=False)
        vggt_elapsed = time.perf_counter() - vggt_start
        chunk_depth_elapsed: Optional[float] = None
        chunk_depth_success = 0
        LOGGER.info(
            "Chunk %d | VGGT processed %d frames in %.2fs (%.2f FPS)",
            chunk_index,
            chunk_frames,
            vggt_elapsed,
            chunk_frames / max(vggt_elapsed, 1e-6),
        )

        depth_raw = preds["depth"].squeeze(0)
        if depth_raw.dim() == 4 and depth_raw.shape[-1] == 1:
            depth_scalar = depth_raw[..., 0]
        elif depth_raw.dim() == 4 and depth_raw.shape[1] == 1:
            depth_scalar = depth_raw[:, 0]
        else:
            depth_scalar = depth_raw

        depth_conf_tensor = preds["depth_conf"].squeeze(0)
        if depth_conf_tensor.dim() == 4 and depth_conf_tensor.shape[-1] == 1:
            depth_conf = depth_conf_tensor[..., 0]
        elif depth_conf_tensor.dim() == 4 and depth_conf_tensor.shape[1] == 1:
            depth_conf = depth_conf_tensor[:, 0]
        else:
            depth_conf = depth_conf_tensor

        pose_enc = preds["pose_enc"]
        image_hw = tuple(chunk.tensor.shape[-2:])
        extrinsic_list: List[torch.Tensor] = []
        intrinsic_list: List[torch.Tensor] = []
        for local_idx in range(pose_enc.shape[1]):
            pose_slice = pose_enc[:, local_idx : local_idx + 1, :].contiguous()
            extrinsic_single, intrinsic_single = pose_encoding_to_extri_intri(pose_slice, image_hw)
            extrinsic_list.append(extrinsic_single[0, 0])
            intrinsic_list.append(intrinsic_single[0, 0])
        extrinsic = torch.stack(extrinsic_list, dim=0)
        intrinsic = torch.stack(intrinsic_list, dim=0)
        rgb_images = preds["images"].squeeze(0)

        use_optimized = optimized_view is not None
        if use_optimized:
            mask = depth_conf > args.depth_conf_threshold
        else:
            point_map_unproj, depth_valid_mask = gpu_unproject_depth_map(
                depth_scalar, extrinsic, intrinsic
            )
            mask = depth_valid_mask & (depth_conf > args.depth_conf_threshold)

        extrinsic_cpu = extrinsic.detach().cpu().numpy()
        intrinsic_cpu = intrinsic.detach().cpu().numpy()
        frame_offset = len(metadata["frame_indices"])

        for local_idx, frame_info in enumerate(chunk.info):
            entry_index = frame_offset + local_idx
            metadata["frame_indices"].append(int(frame_info.index))
            metadata["frame_cameras"].append(frame_info.camera)
            metadata["frame_camera_indices"].append(int(frame_info.camera_index))
            metadata["extrinsics"].append(extrinsic_cpu[local_idx].tolist())
            metadata["intrinsics"].append(intrinsic_cpu[local_idx].tolist())
            metadata["depth_anything_scale"].append({"scale": None, "shift": None})
            metadata["depth_anything_raw_min"].append(None)
            metadata["depth_anything_raw_max"].append(None)
            metadata["depth_anything_rect_min"].append(None)
            metadata["depth_anything_rect_max"].append(None)
            metadata["depth_anything_mae"].append(None)
            metadata["depth_anything_rmse"].append(None)

            mask_frame = mask[local_idx]
            valid_pixels = int(mask_frame.sum().item())
            if valid_pixels == 0:
                metadata["vggt_depth_min"].append(None)
                metadata["vggt_depth_max"].append(None)
                LOGGER.debug(
                    "Chunk %d | VGGT produced no valid points for frame %d",
                    chunk_index,
                    frame_info.index,
                )
                continue

            depth_vals = depth_scalar[local_idx][mask_frame]
            vggt_min = float(depth_vals.min().item())
            vggt_max = float(depth_vals.max().item())
            metadata["vggt_depth_min"].append(vggt_min)
            metadata["vggt_depth_max"].append(vggt_max)
            LOGGER.info(
                "Chunk %d | Frame %d | VGGT depth range: [%0.3f, %0.3f]",
                chunk_index,
                frame_info.index,
                vggt_min,
                vggt_max,
            )

            rgb_tensor = torch.clamp(rgb_images[local_idx], 0, 1)
            rgb_uint8 = (rgb_tensor * 255.0).to(torch.uint8).permute(1, 2, 0)

            if use_optimized:
                depth_np = depth_scalar[local_idx].detach().cpu().numpy()
                depth_conf_np = depth_conf[local_idx].detach().cpu().numpy()
                rgb_np = rgb_tensor.detach().cpu().permute(1, 2, 0).numpy()
                extrinsic_np = extrinsic[local_idx].detach().cpu().numpy()
                intrinsic_np = intrinsic[local_idx].detach().cpu().numpy()
                pts, cols_uint8, reuse_stats = optimized_view.process_frame(
                    frame_info.camera,
                    depth_np,
                    depth_conf_np,
                    rgb_np,
                    extrinsic_np,
                    intrinsic_np,
                )
                chunk_reuse_ratios.append(reuse_stats.reused_ratio)
                chunk_dynamic_pixels += reuse_stats.dynamic_pixels
                chunk_static_pixels += reuse_stats.static_pixels
                cols = cols_uint8
            else:
                pts, cols = _flatten_points(point_map_unproj[local_idx], rgb_uint8, mask_frame)
                if pts.size == 0:
                    LOGGER.debug(
                        "Chunk %d | VGGT produced no valid points for frame %d after flatten.",
                        chunk_index,
                        frame_info.index,
                    )
                    continue

            vggt_point_total += pts.shape[0]
            aggregated_vggt_pts.append(pts)
            aggregated_vggt_cols.append(cols)
            if gaussian_builder is not None and gaussian_mode in {"vggt", "both"}:
                try:
                    conf_weights = depth_conf[local_idx][mask_frame].detach().cpu().numpy()
                except Exception:
                    conf_weights = None
                gaussian_builder.accumulate(
                    pts.astype(np.float32, copy=False),
                    cols.astype(np.float32, copy=False) / 255.0,
                    weights=conf_weights,
                )
            if viz_streamer is not None and args.live_viz_source in {"vggt", "both"}:
                viz_cache = _update_viz_cache(viz_cache, pts.astype(np.float32), cols, args.live_viz_max_points)
            if not first_frame_saved:
                vggt_path = (
                    output_dir
                    / f"{frame_info.camera}_frame{int(frame_info.camera_index):05d}_vggt.ply"
                )
                path_str = None
                if save_ply:
                    write_point_cloud(vggt_path, pts, cols)
                    path_str = str(vggt_path)
                    LOGGER.info("Saved VGGT reference point cloud -> %s", vggt_path)
                else:
                    LOGGER.info("Skipping VGGT reference point cloud save (--no-save-ply)")
                metadata["outputs"].append(
                    {
                        "frame": int(frame_info.index),
                        "camera": frame_info.camera,
                        "camera_frame": int(frame_info.camera_index),
                        "source": "vggt",
                        "path": path_str,
                        "points": int(pts.shape[0]),
                    }
                )
                first_frame_saved = True
                first_frame_idx = int(frame_info.index)
                metadata["depth_reference_frame"] = int(frame_info.index)

        chunk_depth_fps: Optional[float] = None
        chunk_depth_time_ms: Optional[float] = None

        if depth_pool is not None:
            depth_start = time.perf_counter()
            chunk_depth_success = 0
            bgr_inputs = {
                idx: _tensor_to_bgr_uint8(rgb_images[idx]) for idx in range(rgb_images.shape[0])
            }
            depth_da_maps = depth_pool.infer_batch(bgr_inputs)

            if len(depth_da_maps) != chunk_frames:
                LOGGER.warning(
                    "Chunk %d | Depth Anything returned %d maps (expected %d)",
                    chunk_index,
                    len(depth_da_maps),
                    chunk_frames,
                )

            for local_idx, depth_da in sorted(depth_da_maps.items()):
                entry_index = frame_offset + local_idx
                depth_attempted += 1
                depth_da_resized = depth_da
                target_hw = depth_scalar[local_idx].shape
                if depth_da_resized.shape != target_hw:
                    depth_da_resized = cv2.resize(
                        depth_da_resized,
                        (target_hw[1], target_hw[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )

                mask_frame = mask[local_idx]
                if mask_frame.sum().item() < 10:
                    LOGGER.debug(
                        "Chunk %d | Depth Anything mask too small for frame %d",
                        chunk_index,
                        chunk.info[local_idx].index,
                    )
                    continue

                depth_da_tensor = torch.from_numpy(np.ascontiguousarray(depth_da_resized)).to(depth_scalar.device)
                try:
                    scale, shift = _fit_scale_shift(
                        depth_scalar[local_idx],
                        depth_da_tensor,
                        mask_frame,
                        depth_conf[local_idx],
                    )
                except Exception as exc:
                    LOGGER.warning(
                        "Chunk %d | Depth Anything scale fit failed for frame %d: %s",
                        chunk_index,
                        chunk.info[local_idx].index,
                        exc,
                    )
                    continue

                metadata["depth_anything_scale"][entry_index] = {"scale": scale, "shift": shift}
                frame_info = chunk.info[local_idx]
                frame_id = int(frame_info.index)

                rectified_tensor = depth_da_tensor * scale + shift
                raw_vals = depth_da_tensor[mask_frame]
                rect_vals = rectified_tensor[mask_frame]
                raw_min = float(raw_vals.min().item())
                raw_max = float(raw_vals.max().item())
                rect_min = float(rect_vals.min().item())
                rect_max = float(rect_vals.max().item())
                metadata["depth_anything_raw_min"][entry_index] = raw_min
                metadata["depth_anything_raw_max"][entry_index] = raw_max
                metadata["depth_anything_rect_min"][entry_index] = rect_min
                metadata["depth_anything_rect_max"][entry_index] = rect_max

                alignment_err = depth_scalar[local_idx] - rectified_tensor
                alignment_err = alignment_err[mask_frame]
                mae = float(alignment_err.abs().mean().item())
                rmse = float(torch.sqrt(torch.mean(alignment_err ** 2)).item())
                metadata["depth_anything_mae"][entry_index] = mae
                metadata["depth_anything_rmse"][entry_index] = rmse

                if metadata["first_depth_alignment"] is None:
                    metadata["first_depth_alignment"] = {
                        "frame": frame_id,
                        "camera": frame_info.camera,
                        "camera_frame": int(frame_info.camera_index),
                        "scale": scale,
                        "shift": shift,
                        "raw_min": raw_min,
                        "raw_max": raw_max,
                        "rect_min": rect_min,
                        "rect_max": rect_max,
                        "vggt_min": metadata["vggt_depth_min"][entry_index],
                        "vggt_max": metadata["vggt_depth_max"][entry_index],
                        "mae": mae,
                        "rmse": rmse,
                    }

                LOGGER.info(
                    "Chunk %d | Frame %d depth stats | VGGT[%0.3f,%0.3f] | DA raw[%0.3f,%0.3f] → rect[%0.3f,%0.3f] "
                    "| MAE=%.4f | RMSE=%.4f",
                    chunk_index,
                    chunk.info[local_idx].index,
                    metadata["vggt_depth_min"][entry_index],
                    metadata["vggt_depth_max"][entry_index],
                    raw_min,
                    raw_max,
                    rect_min,
                    rect_max,
                    mae,
                    rmse,
                )

                rectified_tensor = rectified_tensor.unsqueeze(0)
                point_map_da, da_valid_mask = gpu_unproject_depth_map(
                    rectified_tensor,
                    extrinsic[local_idx : local_idx + 1],
                    intrinsic[local_idx : local_idx + 1],
                )
                da_mask = da_valid_mask.squeeze(0) & mask_frame
                if da_mask.sum().item() == 0:
                    LOGGER.debug(
                        "Chunk %d | Depth Anything produced empty mask for frame %d",
                        chunk_index,
                        chunk.info[local_idx].index,
                    )
                    continue

                rgb_uint8 = (torch.clamp(rgb_images[local_idx], 0, 1) * 255.0).to(torch.uint8).permute(1, 2, 0)
                pts, cols = _flatten_points(point_map_da.squeeze(0), rgb_uint8, da_mask)
                if pts.size == 0:
                    LOGGER.debug(
                        "Chunk %d | Depth Anything produced no valid points for frame %d",
                        chunk_index,
                        chunk.info[local_idx].index,
                    )
                    continue

                if first_frame_idx is not None and frame_id == first_frame_idx:
                    LOGGER.debug("Skipping depth PLY for reference frame %d", frame_id)
                    continue

                depth_path = (
                    output_dir
                    / f"{frame_info.camera}_frame{int(frame_info.camera_index):05d}_depth_anything.ply"
                )
                depth_path_str = None
                if save_ply:
                    write_point_cloud(depth_path, pts.astype(np.float32), cols)
                    depth_path_str = str(depth_path)
                    LOGGER.info(
                        "Saved Depth Anything point cloud -> %s (points=%d, scale=%.4f, shift=%.4f)",
                        depth_path,
                        pts.shape[0],
                        scale,
                        shift,
                    )
                else:
                    LOGGER.info(
                        "Depth Anything point cloud generated (points=%d, scale=%.4f, shift=%.4f) [no-save]",
                        pts.shape[0],
                        scale,
                        shift,
                    )
                aggregated_depth_pts.append(pts.astype(np.float32))
                aggregated_depth_cols.append(cols)
                depth_point_total += pts.shape[0]
                metadata["outputs"].append(
                    {
                        "frame": frame_id,
                        "camera": frame_info.camera,
                        "camera_frame": int(frame_info.camera_index),
                        "source": "depth_anything",
                        "path": depth_path_str,
                        "points": int(pts.shape[0]),
                        "scale": scale,
                        "shift": shift,
                        "mae": mae,
                        "rmse": rmse,
                    }
                )

                if gaussian_builder is not None and gaussian_mode in {"depth", "both"}:
                    try:
                        depth_weights = depth_conf[local_idx][da_mask].detach().cpu().numpy()
                    except Exception:
                        depth_weights = None
                    gaussian_builder.accumulate(
                        pts.astype(np.float32, copy=False),
                        cols.astype(np.float32, copy=False) / 255.0,
                        weights=depth_weights,
                    )

                if viz_streamer is not None and args.live_viz_source in {"depth", "both"}:
                    viz_cache = _update_viz_cache(viz_cache, pts.astype(np.float32), cols, args.live_viz_max_points)

                depth_success += 1
                chunk_depth_success += 1

            chunk_depth_elapsed = time.perf_counter() - depth_start
            if chunk_depth_success > 0:
                chunk_depth_fps = chunk_depth_success / max(chunk_depth_elapsed, 1e-6)
                chunk_depth_time_ms = chunk_depth_elapsed * 1e3

        vggt_fps = chunk_frames / max(vggt_elapsed, 1e-6)
        summary = (
            f"Chunk {chunk_index:03d} | VGGT {chunk_frames}f → {vggt_elapsed * 1e3:.1f} ms ({vggt_fps:.1f} FPS)"
        )
        if depth_pool is not None:
            if chunk_depth_elapsed is not None and chunk_depth_success > 0 and chunk_depth_time_ms is not None:
                summary += (
                    f" | Depth {chunk_depth_success}f → {chunk_depth_time_ms:.1f} ms "
                    f"({chunk_depth_fps:.1f} FPS)"
                )
            else:
                summary += " | Depth n/a"
        LOGGER.info(summary)

        if use_optimized and chunk_reuse_ratios:
            reuse_mean = sum(chunk_reuse_ratios) / len(chunk_reuse_ratios)
            optimized_chunk_log.append(
                {
                    "chunk": chunk_index,
                    "reuse_mean": reuse_mean,
                    "dynamic_pixels": chunk_dynamic_pixels,
                    "static_pixels": chunk_static_pixels,
                }
            )
            LOGGER.info(
                "Chunk %03d | Optimized reuse=%.1f%% | dynamic=%d px | static=%d px",
                chunk_index,
                reuse_mean * 100.0,
                chunk_dynamic_pixels,
                chunk_static_pixels,
            )

        chunk_metrics_list.append(
            {
                "chunk": chunk_index,
                "frames": chunk_frames,
                "vggt_ms": vggt_elapsed * 1e3,
                "vggt_fps": vggt_fps,
                "depth_ms": chunk_depth_time_ms,
                "depth_fps": chunk_depth_fps,
                "depth_frames": chunk_depth_success,
            }
        )

        if viz_streamer is not None and viz_cache is not None and (chunk_index % viz_interval == 0):
            viz_streamer.push(viz_cache[0], viz_cache[1])

    total_elapsed = time.perf_counter() - total_start
    metadata["depth_attempted"] = depth_attempted
    metadata["depth_success"] = depth_success
    metadata["vggt_point_count"] = int(vggt_point_total)
    metadata["depth_point_count"] = int(depth_point_total)
    metadata["runtime_seconds"] = total_elapsed

    if not first_frame_saved:
        LOGGER.warning("No VGGT reference point cloud was saved.")
    if depth_pool is not None and depth_success == 0:
        LOGGER.warning("Depth Anything produced no point clouds.")

    if aggregated_vggt_pts and save_ply:
        vggt_points_all = np.concatenate(aggregated_vggt_pts, axis=0)
        vggt_colors_all = np.concatenate(aggregated_vggt_cols, axis=0)
        vggt_agg_path = output_dir / "vggt_point_cloud_aggregate.ply"
        LOGGER.info("Saving aggregated VGGT point cloud (%d points)-> %s", vggt_points_all.shape[0], vggt_agg_path)
        write_point_cloud(vggt_agg_path, vggt_points_all, vggt_colors_all)
        metadata["outputs"].append(
            {
                "frame": int(first_frame_idx) if first_frame_idx is not None else None,
                "camera": "aggregate",
                "camera_frame": None,
                "source": "vggt_aggregate",
                "path": str(vggt_agg_path),
                "points": int(vggt_points_all.shape[0]),
            }
        )
    elif aggregated_vggt_pts:
        LOGGER.info(
            "Aggregated VGGT point cloud (%d points) not written (--no-save-ply)",
            np.concatenate(aggregated_vggt_pts, axis=0).shape[0],
        )

    if aggregated_depth_pts and save_ply:
        depth_points_all = np.concatenate(aggregated_depth_pts, axis=0)
        depth_colors_all = np.concatenate(aggregated_depth_cols, axis=0)
        depth_agg_path = output_dir / "depth_anything_point_cloud_aggregate.ply"
        LOGGER.info(
            "Saving aggregated Depth Anything point cloud (%d points)-> %s",
            depth_points_all.shape[0],
            depth_agg_path,
        )
        write_point_cloud(depth_agg_path, depth_points_all, depth_colors_all)
        metadata["outputs"].append(
            {
                "frame": None,
                "camera": "aggregate",
                "camera_frame": None,
                "source": "depth_anything_aggregate",
                "path": str(depth_agg_path),
                "points": int(depth_points_all.shape[0]),
            }
        )
    elif aggregated_depth_pts:
        LOGGER.info(
            "Aggregated Depth Anything point cloud (%d points) not written (--no-save-ply)",
            np.concatenate(aggregated_depth_pts, axis=0).shape[0],
        )

    gaussian_summary = None
    if gaussian_builder is not None:
        gaussian_data = gaussian_builder.to_gaussians()
        gaussian_count = int(gaussian_data["means"].shape[0])
        gaussian_path = output_dir / "gaussians_init.npz"
        np.savez(gaussian_path, **gaussian_data)
        metadata["gaussian_count"] = gaussian_count
        metadata["gaussian_path"] = str(gaussian_path)
        gaussian_summary = {
            "count": gaussian_count,
            "voxel_size": float(args.gaussian_voxel_size),
            "min_weight": float(args.gaussian_min_points),
            "base_variance": float(args.gaussian_base_variance),
        }
        LOGGER.info(
            "Gaussian field initialised | count=%d | voxel=%.4f | output=%s",
            gaussian_count,
            args.gaussian_voxel_size,
            gaussian_path,
        )
    else:
        metadata["gaussian_count"] = 0

    chunk_metrics = metadata["chunk_metrics"]
    if chunk_metrics:
        vggt_ms = [float(cm["vggt_ms"]) for cm in chunk_metrics if cm.get("vggt_ms") is not None]
    else:
        vggt_ms = []
    vggt_total_ms = float(sum(vggt_ms)) if vggt_ms else 0.0
    vggt_total_s = vggt_total_ms / 1e3 if vggt_ms else 0.0
    vggt_mean_fps = frames_processed / max(vggt_total_s, 1e-6) if vggt_ms else 0.0
    vggt_median_ms = float(stats.median(vggt_ms)) if vggt_ms else None
    vggt_min_ms = float(min(vggt_ms)) if vggt_ms else None
    vggt_max_ms = float(max(vggt_ms)) if vggt_ms else None

    depth_ms_list = [float(cm["depth_ms"]) for cm in chunk_metrics if cm.get("depth_ms") is not None]
    depth_total_ms = float(sum(depth_ms_list)) if depth_ms_list else 0.0
    depth_total_s = depth_total_ms / 1e3 if depth_ms_list else 0.0
    depth_frames_total = sum(int(cm["depth_frames"] or 0) for cm in chunk_metrics)
    depth_mean_fps = (
        depth_frames_total / max(depth_total_s, 1e-6) if depth_ms_list and depth_frames_total > 0 else None
    )
    depth_median_ms = float(stats.median(depth_ms_list)) if depth_ms_list else None

    mae_values = [float(v) for v in metadata["depth_anything_mae"] if v is not None]
    rmse_values = [float(v) for v in metadata["depth_anything_rmse"] if v is not None]
    mae_mean = sum(mae_values) / len(mae_values) if mae_values else None
    mae_median = float(stats.median(mae_values)) if mae_values else None
    rmse_mean = sum(rmse_values) / len(rmse_values) if rmse_values else None
    rmse_median = float(stats.median(rmse_values)) if rmse_values else None

    aggregate_metrics = {
        "frames_processed": int(frames_processed),
        "chunks": len(chunk_metrics),
        "runtime_seconds": total_elapsed,
        "vggt": {
            "total_time_seconds": vggt_total_s if vggt_ms else None,
            "mean_fps": vggt_mean_fps if vggt_ms else None,
            "median_ms": vggt_median_ms,
            "min_ms": vggt_min_ms,
            "max_ms": vggt_max_ms,
        },
        "depth_anything": {
            "total_time_seconds": depth_total_s if depth_ms_list else None,
            "mean_fps": depth_mean_fps,
            "median_ms": depth_median_ms,
            "frames": depth_frames_total,
            "success": depth_success,
            "attempted": depth_attempted,
            "mae_mean": mae_mean,
            "mae_median": mae_median,
            "rmse_mean": rmse_mean,
            "rmse_median": rmse_median,
        },
    }
    if gaussian_summary is not None:
        aggregate_metrics["gaussian"] = gaussian_summary

    metadata["aggregate_metrics"] = aggregate_metrics

    LOGGER.info("=== Aggregate Metrics ===")
    LOGGER.info(
        "VGGT | frames=%d | total=%.2fs | avg=%s | median=%s ms | min=%s ms | max=%s ms",
        frames_processed,
        vggt_total_s,
        f"{vggt_mean_fps:.1f} FPS" if vggt_ms else "n/a",
        f"{vggt_median_ms:.1f}" if vggt_median_ms is not None else "n/a",
        f"{vggt_min_ms:.1f}" if vggt_min_ms is not None else "n/a",
        f"{vggt_max_ms:.1f}" if vggt_max_ms is not None else "n/a",
    )
    if depth_ms_list:
        LOGGER.info(
            "Depth Anything | frames=%d | total=%.2fs | avg=%s | median=%s ms | MAE(avg=%s, med=%s) | "
            "RMSE(avg=%s, med=%s) | success=%d/%d",
            depth_frames_total,
            depth_total_s,
            f"{depth_mean_fps:.1f} FPS" if depth_mean_fps is not None else "n/a",
            f"{depth_median_ms:.1f}" if depth_median_ms is not None else "n/a",
            f"{mae_mean:.4f}" if mae_mean is not None else "n/a",
            f"{mae_median:.4f}" if mae_median is not None else "n/a",
            f"{rmse_mean:.4f}" if rmse_mean is not None else "n/a",
            f"{rmse_median:.4f}" if rmse_median is not None else "n/a",
            depth_success,
            depth_attempted,
        )
    else:
        LOGGER.info(
            "Depth Anything | frames=0 | total=0.00s | avg=n/a | median=n/a | MAE(n/a, n/a) | RMSE(n/a, n/a) | success=%d/%d",
            depth_success,
            depth_attempted,
        )
    if metadata["first_depth_alignment"] is not None:
        align = metadata["first_depth_alignment"]
        LOGGER.info(
            "First depth alignment | frame=%d (%s) | scale=%.5f | shift=%.5f | VGGT=[%.3f, %.3f] | "
            "Depth raw=[%.3f, %.3f] → rect=[%.3f, %.3f] | MAE=%.4f | RMSE=%.4f",
            align["frame"],
            align["camera"],
            align["scale"],
            align["shift"],
            align["vggt_min"],
            align["vggt_max"],
            align["raw_min"],
            align["raw_max"],
            align["rect_min"],
            align["rect_max"],
            align.get("mae", float("nan")),
            align.get("rmse", float("nan")),
        )

    if viz_streamer is not None and viz_cache is not None:
        viz_streamer.push(viz_cache[0], viz_cache[1])

    metadata_path = output_dir / "metadata.json"
    if optimized_chunk_log:
        metadata["optimized_chunks"] = optimized_chunk_log
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    LOGGER.info("Metadata written to %s", metadata_path)

    LOGGER.info(
        "Pipeline complete: frames=%d | runtime=%.2fs | VGGT points=%d | Depth points=%d | Depth success=%d/%d",
        frames_processed,
        total_elapsed,
        vggt_point_total,
        depth_point_total,
        depth_success,
        depth_attempted,
    )

    if depth_pool is not None:
        depth_pool.close()

    if viz_streamer is not None:
        viz_streamer.close()


def write_point_cloud(path: Path, points: np.ndarray, colors: Optional[np.ndarray] = None) -> None:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be of shape (N, 3).")
    num_points = points.shape[0]
    path = Path(path)
    with open(path, "w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {num_points}\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            colors = np.asarray(colors, dtype=np.uint8)
            if colors.shape != (num_points, 3):
                raise ValueError("colors must be of shape (N, 3).")
            file.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        file.write("end_header\n")
        if colors is None:
            for pt in points:
                file.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")
        else:
            for pt, col in zip(points, colors):
                file.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} {int(col[0])} {int(col[1])} {int(col[2])}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simplified VGGT + Depth Anything point cloud reconstruction.")
    parser.add_argument(
        "--input-type",
        choices=["images", "multicam", "video"],
        default="images",
        help="Use 'images' for a flat frame directory, 'multicam' for per-camera folders/prefixes, or 'video' for file inputs.",
    )
    parser.add_argument("--path", required=True, help="Input directory (for images) or video file path.")
    parser.add_argument("--output-dir", default="pcd_out/simple", help="Directory for generated outputs.")
    parser.add_argument("--vggt-weights", default="vggt_model.pt", help="Path to VGGT PyTorch weights.")
    parser.add_argument(
        "--depth-anything",
        choices=["auto", "off"],
        default="auto",
        help="Enable Depth Anything rectification. Use 'off' to skip.",
    )
    parser.add_argument(
        "--depth-anything-engine",
        default="onnx_exports/depth_anything",
        help="Directory (or explicit path) to Depth Anything exports (.onnx or .engine).",
    )
    parser.add_argument(
        "--depth-backend",
        choices=["auto", "tensorrt", "onnxruntime"],
        default="auto",
        help="Depth Anything inference backend. 'auto' prefers TensorRT when available.",
    )
    parser.add_argument(
        "--depth-workers",
        type=int,
        default=0,
        help="Number of Depth Anything workers (0=auto).",
    )
    parser.add_argument("--device", default=None, help="Torch device override, e.g. 'cuda:0' or 'cpu'.")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit the number of processed frames.")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of frames per VGGT forward pass.")
    parser.add_argument(
        "--image-loops",
        type=int,
        default=1,
        help="Repeat the image sequence this many times to simulate a longer stream.",
    )
    parser.add_argument(
        "--depth-conf-threshold",
        type=float,
        default=0.3,
        help="Minimum VGGT depth confidence to keep a pixel.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level for the pipeline output.",
    )
    parser.add_argument(
        "--no-save-ply",
        action="store_true",
        help="Skip writing per-frame and aggregate PLY files (useful for pure throughput tests).",
    )
    parser.add_argument(
        "--live-viz",
        choices=["none", "o3d"],
        default="none",
        help="Enable live point-cloud visualization (requires open3d).",
    )
    parser.add_argument(
        "--live-viz-source",
        choices=["vggt", "depth", "both"],
        default="vggt",
        help="Select which point clouds to stream into the live viewer.",
    )
    parser.add_argument(
        "--live-viz-max-points",
        type=int,
        default=200_000,
        help="Maximum number of points retained in the live viewer buffer.",
    )
    parser.add_argument(
        "--live-viz-interval",
        type=int,
        default=1,
        help="Update the live viewer every N processed chunks.",
    )
    parser.add_argument(
        "--live-viz-point-size",
        type=int,
        default=1,
        help="Point size used by the live viewer.",
    )
    parser.add_argument(
        "--gaussian-init",
        choices=["none", "vggt", "depth", "both"],
        default="none",
        help="Generate Gaussian splats from point clouds.",
    )
    parser.add_argument(
        "--gaussian-voxel-size",
        type=float,
        default=0.01,
        help="Voxel size (in world units) used for Gaussian aggregation.",
    )
    parser.add_argument(
        "--gaussian-min-points",
        type=float,
        default=10.0,
        help="Minimum accumulated weight per voxel before emitting a Gaussian.",
    )
    parser.add_argument(
        "--gaussian-base-variance",
        type=float,
        default=1e-4,
        help="Floor variance to keep Gaussians well-conditioned.",
    )
    parser.add_argument(
        "--gaussian-max-count",
        type=int,
        default=250_000,
        help="Maximum number of Gaussians retained (0 = unlimited).",
    )
    parser.add_argument(
        "--optimized-live-view",
        action="store_true",
        help="Enable geometry reuse pipeline for static scenes.",
    )
    parser.add_argument(
        "--optimized-depth-threshold",
        type=float,
        default=0.02,
        help="Depth residual threshold (meters) for detecting geometry change.",
    )
    parser.add_argument(
        "--optimized-color-threshold",
        type=float,
        default=0.12,
        help="Color residual threshold (L2 in [0,1]) for detecting appearance change.",
    )
    parser.add_argument(
        "--optimized-confidence-threshold",
        type=float,
        default=0.25,
        help="Minimum VGGT depth confidence for reuse decisions.",
    )
    parser.add_argument(
        "--optimized-ema",
        type=float,
        default=0.05,
        help="Exponential moving average decay for template updates.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    level = getattr(logging, args.log_level.upper(), logging.INFO)
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    else:
        root_logger.setLevel(level)
    LOGGER.setLevel(level)
    logging.getLogger("reconstruction.depth_anything").setLevel(level)
    run_pipeline(args)


if __name__ == "__main__":
    main()
