"""
Simple raycasting utilities for expected depth generation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def project_points(
    points_w: np.ndarray,
    pose_wc: np.ndarray,
    intrinsics: np.ndarray,
    image_size: Tuple[int, int],
    z_buffer: Optional[np.ndarray] = None,
) -> np.ndarray:
    h, w = image_size
    depth = np.full((h, w), np.nan, dtype=np.float32) if z_buffer is None else z_buffer
    pose_cw = np.linalg.inv(pose_wc)
    homog = np.concatenate([points_w, np.ones((points_w.shape[0], 1), dtype=np.float32)], axis=1)
    pts_c = (pose_cw @ homog.T).T[:, :3]
    valid = pts_c[:, 2] > 1e-6
    pts_c = pts_c[valid]
    if pts_c.size == 0:
        return depth
    pixels = (intrinsics @ pts_c.T).T
    pixels = pixels[:, :2] / pixels[:, 2:3]
    u = np.round(pixels[:, 0]).astype(int)
    v = np.round(pixels[:, 1]).astype(int)
    mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z = u[mask], v[mask], pts_c[:, 2][mask]
    if u.size == 0:
        return depth
    flat = v * w + u
    current = depth.reshape(-1)
    existing = current[flat]
    update = np.isnan(existing) | (z < existing)
    current[flat[update]] = z[update]
    depth[:] = current.reshape(h, w)
    return depth


def expected_depth_from_surfel(
    points_w: np.ndarray,
    pose_wc: np.ndarray,
    intrinsics: np.ndarray,
    image_size: Tuple[int, int],
) -> np.ndarray:
    return project_points(points_w, pose_wc, intrinsics, image_size)


@dataclass
class TSDFGrid:
    origin: np.ndarray
    voxel_size: float
    sdf: np.ndarray
    weights: np.ndarray

    def raycast(
        self,
        pose_wc: np.ndarray,
        intrinsics: np.ndarray,
        image_size: Tuple[int, int],
        max_steps: int = 64,
        trunc_margin: float = 0.04,
    ) -> np.ndarray:
        h, w = image_size
        depth = np.full((h, w), np.nan, dtype=np.float32)
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        grid_shape = np.array(self.sdf.shape[::-1], dtype=np.float32)
        origin_w = pose_wc[:3, 3]
        R = pose_wc[:3, :3]
        for v in range(h):
            for u in range(w):
                dir_c = np.array([(u - cx) / fx, (v - cy) / fy, 1.0], dtype=np.float32)
                dir_c /= np.linalg.norm(dir_c)
                dir_w = R @ dir_c
                dir_w /= np.linalg.norm(dir_w)
                t = 0.0
                last_sdf = None
                for _ in range(max_steps):
                    point_w = origin_w + dir_w * t
                    voxel = (point_w - self.origin) / self.voxel_size
                    if np.any(voxel < 0) or np.any(voxel >= grid_shape):
                        break
                    vi = np.floor(voxel).astype(int)
                    sdf_val = self.sdf[vi[2], vi[1], vi[0]]
                    if last_sdf is not None and last_sdf > 0 >= sdf_val:
                        depth[v, u] = t
                        break
                    last_sdf = sdf_val
                    t += max(sdf_val, trunc_margin / 2)
        return depth

