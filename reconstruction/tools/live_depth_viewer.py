#!/usr/bin/env python3
"""
Live point-cloud visualizer for Depth Anything v2 outputs.

The script streams images from a directory, runs Depth Anything v2 (TensorRT or
ONNX Runtime via :class:`DepthAnythingPool`), unprojects the depth into a metric
point cloud using user-specified intrinsics, and visualizes the result with
Open3D.  The viewer updates the same geometry in-place so it can keep up with
live feeds without recreating the entire scene.

Example:
    python -m reconstruction.tools.live_depth_viewer \\
        --images cam_snaps/left \\
        --engine onnx_exports/depth_anything/depth_anything_v2_vitl.engine \\
        --fx 910.0 --fy 910.0 --cx 512.0 --cy 512.0
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple

import numpy as np

try:
    import cv2
except Exception as exc:  # pragma: no cover
    raise RuntimeError("OpenCV is required. Install with `pip install opencv-python`.") from exc

try:
    import open3d as o3d
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Open3D is required. Install with `pip install open3d`.") from exc

from reconstruction.pcd.depth_anything import DepthAnythingPool


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(folder: Path) -> Iterable[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def unproject_depth(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Back-project a depth map into 3D camera coordinates."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    z = depth.reshape(-1)
    valid = np.isfinite(z) & (z > 0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)

    x = (u.reshape(-1) - cx) / fx * z
    y = (v.reshape(-1) - cy) / fy * z

    points = np.stack([x, y, z], axis=1)[valid]
    return points.astype(np.float32, copy=False)


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def colorize_rgb(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8 RGB to float colors in [0, 1]."""
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8, copy=False)
    return (rgb.reshape(-1, 3).astype(np.float32) / 255.0)


def iter_images(paths: Iterable[Path], loop: bool) -> Iterator[Tuple[int, Path]]:
    paths = list(paths)
    if not paths:
        raise RuntimeError("No images found for streaming.")
    index = 0
    while True:
        for path in paths:
            yield index, path
            index += 1
        if not loop:
            break


def create_intrinsics(args: argparse.Namespace) -> np.ndarray:
    fx = float(args.fx)
    fy = float(args.fy)
    cx = float(args.cx)
    cy = float(args.cy)
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream Depth Anything v2 point clouds with live visualization.",
    )
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Directory containing source RGB frames.",
    )
    parser.add_argument(
        "--engine",
        type=Path,
        required=True,
        help="Path to Depth Anything TensorRT engine or ONNX directory.",
    )
    parser.add_argument(
        "--intrinsics",
        type=Path,
        help="Optional npz/json file with 3x3 camera matrix (expects key 'K').",
    )
    parser.add_argument("--fx", type=float, help="Focal length in pixels (x-axis).")
    parser.add_argument("--fy", type=float, help="Focal length in pixels (y-axis).")
    parser.add_argument("--cx", type=float, help="Principal point x-coordinate in pixels.")
    parser.add_argument("--cy", type=float, help="Principal point y-coordinate in pixels.")
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop over the image directory indefinitely.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.0,
        help="Optional voxel down-sampling size in meters.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="Delay (seconds) between frames to simulate live streaming.",
    )
    parser.add_argument(
        "--precision",
        default="auto",
        choices=["auto", "fp32", "fp16", "bf16", "int8"],
        help="Preferred engine precision if multiple variants are present.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of Depth Anything workers (defaults to 1).",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    image_dir = args.images.expanduser().resolve()
    if not image_dir.is_dir():
        raise SystemExit(f"[ERROR] Image directory not found: {image_dir}")

    if args.intrinsics:
        intrinsics = np.load(args.intrinsics)["K"]  # type: ignore[index]
    else:
        missing = [name for name in ("fx", "fy", "cx", "cy") if getattr(args, name) is None]
        if missing:
            raise SystemExit(f"[ERROR] Provide --intrinsics or specify {', '.join(missing)}.")
        intrinsics = create_intrinsics(args)

    image_paths = list_images(image_dir)
    if not image_paths:
        raise SystemExit(f"[ERROR] No images found under {image_dir}")

    vis = o3d.visualization.Visualizer()
    vis.create_window("Depth Anything Live Viewer", width=1280, height=960, visible=True)
    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    engine_path = args.engine.expanduser().resolve()
    pool_kwargs: Dict[str, object] = {
        "precision": args.precision,
        "num_workers": max(1, int(args.num_workers)),
    }
    if engine_path.is_file():
        pool_kwargs["explicit_engine"] = engine_path
        engine_dir = engine_path.parent
    else:
        engine_dir = engine_path

    with DepthAnythingPool(engine_dir, **pool_kwargs) as pool:
        for index, image_path in iter_images(image_paths, loop=args.loop):
            rgb = load_rgb(image_path)
            h, w, _ = rgb.shape

            # Depth inference
            depth_maps = pool.infer_batch({image_path.stem: rgb})
            depth = depth_maps[image_path.stem]
            if depth.shape != (h, w):
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

            points = unproject_depth(depth, intrinsics)
            if points.size == 0:
                print(f"[WARN] Frame {index}: no valid points produced.")
                continue

            colors = colorize_rgb(rgb)
            colors = colors[: points.shape[0]]

            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            if args.voxel_size > 0.0:
                pcd = pcd.voxel_down_sample(voxel_size=float(args.voxel_size))

            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            print(f"[INFO] Frame {index:04d} | points={points.shape[0]:,}")
            if args.interval > 0.0:
                time.sleep(args.interval)
    vis.destroy_window()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
