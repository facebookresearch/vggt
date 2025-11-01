"""
Optimized live-view helper that reuses cached geometry for static regions.

This module keeps all computations analytical so we rely only on VGGT / Depth
Anything outputs.  It detects geometric/photometric residuals, reprojects cached
points for stable pixels, and only recomputes world coordinates where motion is
detected.  The result is a blended point set plus statistics for logging.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class CameraTemplate:
    rays_cam: np.ndarray  # (H, W, 3) ray directions in camera space
    depth: np.ndarray  # (H, W)
    color: np.ndarray  # (H, W, 3) float32 in [0,1]
    world_points: np.ndarray  # (H, W, 3)
    confidence: np.ndarray  # (H, W) running estimate
    initialized: bool = False


@dataclass
class FrameReuseStats:
    reused_ratio: float
    dynamic_pixels: int
    static_pixels: int
    depth_residual_mean: float
    color_residual_mean: float


class OptimizedLiveView:
    """
    Maintains per-camera templates and provides blended point clouds that reuse
    cached geometry whenever possible.
    """

    def __init__(
        self,
        *,
        depth_threshold: float = 0.02,
        color_threshold: float = 0.12,
        confidence_threshold: float = 0.25,
        ema_decay: float = 0.05,
        max_points_per_frame: int = 150000,
    ) -> None:
        self.depth_thresh = float(depth_threshold)
        self.color_thresh = float(color_threshold)
        self.conf_thresh = float(confidence_threshold)
        self.ema_decay = float(np.clip(ema_decay, 0.0, 1.0))
        self.max_points = int(max_points_per_frame)
        self.templates: Dict[str, CameraTemplate] = {}

    @staticmethod
    def _compute_rays(intrinsic: np.ndarray, height: int, width: int) -> np.ndarray:
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]
        u = np.arange(width, dtype=np.float32)
        v = np.arange(height, dtype=np.float32)
        grid_u, grid_v = np.meshgrid(u, v)
        x = (grid_u - cx) / fx
        y = (grid_v - cy) / fy
        ones = np.ones_like(x)
        rays = np.stack([x, y, ones], axis=-1)
        return rays.astype(np.float32)

    @staticmethod
    def _cam_to_world(points_cam: np.ndarray, extrinsic: np.ndarray) -> np.ndarray:
        R = extrinsic[:3, :3]
        t = extrinsic[:3, 3]
        R_inv = R.T
        t_inv = -R_inv @ t
        points = points_cam @ R_inv.T + t_inv
        return points

    def _ensure_template(
        self,
        camera: str,
        depth: np.ndarray,
        color_rgb: np.ndarray,
        depth_conf: np.ndarray,
        extrinsic: np.ndarray,
        intrinsic: np.ndarray,
    ) -> CameraTemplate:
        h, w = depth.shape
        template = self.templates.get(camera)
        if template is None:
            rays = self._compute_rays(intrinsic, h, w)
            xyz_cam = rays * depth[..., None]
            world = self._cam_to_world(xyz_cam.reshape(-1, 3), extrinsic).reshape(h, w, 3)
            template = CameraTemplate(
                rays_cam=rays,
                depth=depth.copy(),
                color=color_rgb.copy(),
                world_points=world.astype(np.float32, copy=False),
                confidence=depth_conf.copy(),
                initialized=True,
            )
            self.templates[camera] = template
        elif not template.initialized:
            template.depth[...] = depth
            template.color[...] = color_rgb
            xyz_cam = template.rays_cam * depth[..., None]
            template.world_points[...] = self._cam_to_world(
                xyz_cam.reshape(-1, 3), extrinsic
            ).reshape(h, w, 3)
            template.confidence[...] = depth_conf
            template.initialized = True
        return template

    def process_frame(
        self,
        camera: str,
        depth_map: np.ndarray,
        depth_conf: np.ndarray,
        color_rgb: np.ndarray,
        extrinsic: np.ndarray,
        intrinsic: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, FrameReuseStats]:
        """
        Returns (points_world, colors_uint8, stats) for the given view.
        color_rgb should be float32 in [0,1]; caller can convert to uint8 later.
        """
        template = self._ensure_template(
            camera, depth_map, color_rgb, depth_conf, extrinsic, intrinsic
        )
        if template.rays_cam.shape[:2] != depth_map.shape:
            raise ValueError("Image size mismatch for optimized live view.")

        # Compute residual masks
        depth_residual = np.abs(depth_map - template.depth)
        color_residual = np.linalg.norm(color_rgb - template.color, axis=2)
        conf_mask = depth_conf >= self.conf_thresh

        dynamic_mask = (depth_residual > self.depth_thresh) | (color_residual > self.color_thresh)
        dynamic_mask &= conf_mask
        static_mask = conf_mask & (~dynamic_mask)

        if not np.any(dynamic_mask):
            # Entire frame reused; update colors via EMA for stability
            decay = self.ema_decay
            template.color = (1 - decay) * template.color + decay * color_rgb
            template.confidence = np.maximum(template.confidence, depth_conf)
            pts = template.world_points[static_mask]
            cols = np.clip(template.color[static_mask], 0.0, 1.0)
            stats = FrameReuseStats(
                reused_ratio=1.0,
                dynamic_pixels=0,
                static_pixels=int(static_mask.sum()),
                depth_residual_mean=float(depth_residual.mean()),
                color_residual_mean=float(color_residual.mean()),
            )
            return pts, (cols * 255.0).astype(np.uint8), stats

        rays = template.rays_cam
        xyz_cam = (rays * depth_map[..., None]).reshape(-1, 3)
        world_points_all = self._cam_to_world(xyz_cam, extrinsic).reshape(depth_map.shape + (3,))

        # Extract dynamic points
        pts_dynamic = world_points_all[dynamic_mask]
        cols_dynamic = color_rgb[dynamic_mask]

        # Update template with EMA for dynamic regions
        decay = self.ema_decay
        template.depth[dynamic_mask] = (1 - decay) * template.depth[dynamic_mask] + decay * depth_map[dynamic_mask]
        template.color[dynamic_mask] = (1 - decay) * template.color[dynamic_mask] + decay * color_rgb[dynamic_mask]
        template.world_points[dynamic_mask] = (
            (1 - decay) * template.world_points[dynamic_mask]
            + decay * world_points_all[dynamic_mask]
        )
        template.confidence[dynamic_mask] = np.maximum(
            template.confidence[dynamic_mask], depth_conf[dynamic_mask]
        )

        pts_static = template.world_points[static_mask]
        cols_static = template.color[static_mask]

        pts_combined = np.concatenate([pts_static, pts_dynamic], axis=0) if pts_static.size else pts_dynamic
        cols_combined = np.concatenate([cols_static, cols_dynamic], axis=0) if cols_static.size else cols_dynamic

        if pts_combined.shape[0] > self.max_points:
            idx = np.random.choice(pts_combined.shape[0], self.max_points, replace=False)
            pts_combined = pts_combined[idx]
            cols_combined = cols_combined[idx]

        stats = FrameReuseStats(
            reused_ratio=float(static_mask.sum()) / float(static_mask.sum() + dynamic_mask.sum()),
            dynamic_pixels=int(dynamic_mask.sum()),
            static_pixels=int(static_mask.sum()),
            depth_residual_mean=float(depth_residual[conf_mask].mean()) if np.any(conf_mask) else float(depth_residual.mean()),
            color_residual_mean=float(color_residual[conf_mask].mean()) if np.any(conf_mask) else float(color_residual.mean()),
        )
        return pts_combined.astype(np.float32, copy=False), (np.clip(cols_combined, 0.0, 1.0) * 255.0).astype(np.uint8), stats
