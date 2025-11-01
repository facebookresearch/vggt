"""
3D Gaussian field builder for point-cloud derived splats.

This module keeps the implementation purely mathematical so it can operate offline
or in a live setting where new points stream in.  The builder aggregates points into
voxel cells, maintains running statistics, and emits Gaussian parameters that are
compatible with fast Gaussian rasterisers (mean, color, opacity, diagonal covariance).

The intent is to provide a clean bridge between dense point clouds produced by the
VGGT + Depth Anything pipeline and subsequent 3DGS / 4DGS renderers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple

import numpy as np


@dataclass(slots=True)
class _VoxelStats:
    count: float = 0.0
    weight: float = 0.0
    sum_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    sum_pos_sq: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    sum_col: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    sum_col_sq: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def accumulate(self, pts: np.ndarray, cols: np.ndarray, weights: Optional[np.ndarray] = None) -> None:
        if weights is None:
            w = np.ones(pts.shape[0], dtype=np.float64)
        else:
            w = weights.astype(np.float64, copy=False)
        total_w = float(np.sum(w))
        if total_w <= 0.0:
            return
        self.count += float(pts.shape[0])
        self.weight += total_w
        self.sum_pos += np.sum(pts * w[:, None], axis=0)
        self.sum_pos_sq += np.sum((pts ** 2) * w[:, None], axis=0)
        self.sum_col += np.sum(cols * w[:, None], axis=0)
        self.sum_col_sq += np.sum((cols ** 2) * w[:, None], axis=0)

    def finalize(self, min_weight: float, base_variance: float) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
        if self.weight < min_weight:
            return None
        mean_pos = self.sum_pos / self.weight
        mean_col = np.clip(self.sum_col / self.weight, 0.0, 1.0)
        var_pos = self.sum_pos_sq / self.weight - mean_pos ** 2
        var_pos = np.maximum(var_pos, base_variance)
        opacity = np.clip(self.weight, 0.0, 1.0)
        return mean_pos.astype(np.float32), mean_col.astype(np.float32), var_pos.astype(np.float32), float(opacity)


class GaussianFieldBuilder:
    """
    Aggregates point clouds into a set of Gaussian splats.
    """

    def __init__(
        self,
        *,
        voxel_size: float = 0.01,
        min_weight: float = 10.0,
        base_variance: float = 1e-5,
        max_gaussians: Optional[int] = None,
    ) -> None:
        if voxel_size <= 0:
            raise ValueError("voxel_size must be positive.")
        self.voxel_size = float(voxel_size)
        self.min_weight = float(min_weight)
        self.base_variance = float(base_variance)
        self.max_gaussians = max_gaussians
        self._voxels: Dict[Tuple[int, int, int], _VoxelStats] = {}

    def _voxel_key(self, points: np.ndarray) -> np.ndarray:
        return np.floor(points / self.voxel_size).astype(np.int64)

    def accumulate(
        self,
        points: np.ndarray,
        colors: np.ndarray,
        *,
        weights: Optional[np.ndarray] = None,
        prune_outliers: bool = True,
        max_zscore: float = 4.0,
    ) -> None:
        if points.size == 0:
            return
        if points.shape[0] != colors.shape[0]:
            raise ValueError("points and colors length mismatch.")
        pts = np.asarray(points, dtype=np.float32)
        cols = np.asarray(colors, dtype=np.float32)
        if weights is not None:
            w = np.asarray(weights, dtype=np.float32)
            if w.shape[0] != pts.shape[0]:
                raise ValueError("weights length mismatch.")
        else:
            w = None

        if prune_outliers and pts.shape[0] >= 10:
            mean = np.mean(pts, axis=0)
            std = np.std(pts, axis=0) + 1e-6
            z = np.abs((pts - mean) / std)
            mask = np.all(z < max_zscore, axis=1)
            if np.any(~mask):
                pts = pts[mask]
                cols = cols[mask]
                if w is not None:
                    w = w[mask]
            if pts.size == 0:
                return

        keys = self._voxel_key(pts)
        unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
        for index, (kx, ky, kz) in enumerate(unique_keys):
            mask = inverse == index
            voxel = self._voxels.setdefault((int(kx), int(ky), int(kz)), _VoxelStats())
            voxel.accumulate(pts[mask], cols[mask], w[mask] if w is not None else None)

    def to_gaussians(self) -> Dict[str, np.ndarray]:
        centers: list[np.ndarray] = []
        colors: list[np.ndarray] = []
        covariances: list[np.ndarray] = []
        opacities: list[float] = []

        for voxel in self._voxels.values():
            result = voxel.finalize(self.min_weight, self.base_variance)
            if result is None:
                continue
            mean_pos, mean_col, var_pos, opacity = result
            centers.append(mean_pos)
            colors.append(mean_col)
            covariances.append(var_pos)
            opacities.append(opacity)

        if not centers:
            return {
                "means": np.zeros((0, 3), dtype=np.float32),
                "colors": np.zeros((0, 3), dtype=np.float32),
                "covariances": np.zeros((0, 3), dtype=np.float32),
                "opacities": np.zeros((0,), dtype=np.float32),
            }

        means = np.stack(centers, axis=0)
        cols = np.stack(colors, axis=0)
        covs = np.stack(covariances, axis=0)
        opacities_np = np.asarray(opacities, dtype=np.float32)

        if self.max_gaussians is not None and means.shape[0] > self.max_gaussians:
            energies = np.sum(opacities_np[:, None] * covs, axis=1)
            keep = np.argsort(-energies)[: self.max_gaussians]
            means = means[keep]
            cols = cols[keep]
            covs = covs[keep]
            opacities_np = opacities_np[keep]

        return {
            "means": means,
            "colors": cols,
            "covariances": covs,
            "opacities": opacities_np,
        }

    def reset(self) -> None:
        self._voxels.clear()
