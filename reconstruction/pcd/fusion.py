"""
Surfel and TSDF fusion utilities.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


def backproject_depth(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    pose_wc: np.ndarray,
) -> np.ndarray:
    """
    Back-project a depth map into world-space points.
    """
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    z = depth.reshape(-1)
    valid = np.isfinite(z) & (z > 0)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32)
    x = ((u.reshape(-1) - cx) / fx) * z
    y = ((v.reshape(-1) - cy) / fy) * z
    points_c = np.stack([x, y, z], axis=1)[valid]
    homog = np.concatenate([points_c, np.ones((points_c.shape[0], 1), dtype=np.float32)], axis=1)
    points_w = (pose_wc @ homog.T).T[:, :3]
    return points_w


@dataclass
class SurfelMap:
    """Minimal surfel representation."""

    positions: np.ndarray

    @classmethod
    def from_depth_batch(
        cls,
        depths: Tuple[np.ndarray, ...],
        intrinsics: Tuple[np.ndarray, ...],
        poses_wc: Tuple[np.ndarray, ...],
    ) -> "SurfelMap":
        points = []
        for depth, K, pose in zip(depths, intrinsics, poses_wc):
            pts = backproject_depth(depth, K, pose)
            if pts.size > 0:
                points.append(pts)
        if not points:
            raise ValueError("Initial map produced no valid points.")
        positions = np.concatenate(points, axis=0)
        return cls(positions=positions.astype(np.float32))

    def fuse(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        pose_wc: np.ndarray,
        max_points: int = 1_000_000,
    ) -> None:
        """
        Append new surfels obtained from a metric depth map.
        """
        points = backproject_depth(depth, intrinsics, pose_wc)
        if points.size == 0:
            return
        self.positions = np.concatenate([self.positions, points], axis=0)
        if self.positions.shape[0] > max_points:
            self.positions = self.positions[-max_points:]

    def as_point_cloud(self) -> np.ndarray:
        return self.positions


@dataclass
class TSDFVolume:
    """
    Lightweight TSDF grid for integration/raycasting.
    """

    origin: np.ndarray
    voxel_size: float
    trunc_margin: float
    voxels: Dict[Tuple[int, int, int], Tuple[float, float]] = field(default_factory=dict)

    def _voxel_index(self, point: np.ndarray) -> Tuple[int, int, int]:
        return tuple(np.floor((point - self.origin) / self.voxel_size).astype(int))

    def integrate(
        self,
        depth: np.ndarray,
        intrinsics: np.ndarray,
        pose_wc: np.ndarray,
    ) -> None:
        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
        for i in range(h):
            for j in range(w):
                z = depth[i, j]
                if not np.isfinite(z) or z <= 0:
                    continue
                x = (j - cx) / fx * z
                y = (i - cy) / fy * z
                point_c = np.array([x, y, z, 1.0], dtype=np.float32)
                point_w = (pose_wc @ point_c)[:3]
                voxel = self._voxel_index(point_w)
                voxel_center = self.origin + (np.array(voxel, dtype=np.float32) + 0.5) * self.voxel_size
                ray_origin_w = pose_wc[:3, 3]
                sdf = np.dot(point_w - ray_origin_w, point_w - voxel_center)
                sdf = np.clip(sdf, -self.trunc_margin, self.trunc_margin)
                tsdf = sdf / self.trunc_margin
                prev_val, prev_weight = self.voxels.get(voxel, (0.0, 0.0))
                weight = prev_weight + 1.0
                new_val = (prev_val * prev_weight + tsdf) / weight
                self.voxels[voxel] = (new_val, weight)

    def export(self) -> Tuple[np.ndarray, np.ndarray]:
        coords = []
        values = []
        for idx, (val, _) in self.voxels.items():
            centre = self.origin + (np.array(idx, dtype=np.float32) + 0.5) * self.voxel_size
            coords.append(centre)
            values.append(val)
        if not coords:
            return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.float32)
        return (
            np.stack(coords, axis=0),
            np.asarray(values, dtype=np.float32),
        )

    def to_point_cloud(self, iso_level: float = 0.1) -> np.ndarray:
        coords, values = self.export()
        if coords.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        mask = np.abs(values) <= iso_level
        if not np.any(mask):
            mask = np.abs(values) == np.min(np.abs(values))
        return coords[mask]

    def sample(self, points_w: np.ndarray) -> np.ndarray:
        """Return TSDF values at ``points_w`` (nearest-voxel lookup)."""
        if points_w.size == 0:
            return np.empty(0, dtype=np.float32)
        voxels = np.floor((points_w - self.origin) / self.voxel_size).astype(int)
        values = np.empty(points_w.shape[0], dtype=np.float32)
        values.fill(np.nan)
        for idx, voxel in enumerate(voxels):
            values[idx] = self.voxels.get(tuple(voxel), (np.nan, 0.0))[0]
        return values
