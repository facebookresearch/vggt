"""
Advanced metric depth alignment bridging VGGT bootstrap and Depth Anything.

The goal is to turn Depth Anything's per-frame relative depth predictions into
metric maps by exploiting the existing VGGT reconstruction.  The algorithm
combines weighted least-squares alignment, per-camera Extended Kalman Filters,
and optional TSDF residuals to stabilise the scale across time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np

from .align import AlignmentModel, AlignmentResult


def _apply_model(scale: float, shift: float, rel: np.ndarray, model: AlignmentModel) -> np.ndarray:
    if model is AlignmentModel.SCALE:
        return scale * rel
    if model is AlignmentModel.SCALE_SHIFT:
        return scale * rel + shift
    if model is AlignmentModel.INVDEPTH_SCALE_SHIFT:
        inv = np.reciprocal(np.clip(rel, 1e-6, None), dtype=np.float32)
        metric_inv = scale * inv + shift
        return np.reciprocal(np.clip(metric_inv, 1e-6, None), dtype=np.float32)
    raise ValueError(f"Unsupported alignment model: {model}")


def _weighted_scale_fit(rel: np.ndarray, ref: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    denom = np.dot(weights, rel * rel)
    if denom <= 1e-8:
        raise ValueError("Degenerate weighted scale fit.")
    scale = float(np.dot(weights, rel * ref) / denom)
    return scale, 0.0


def _weighted_scale_shift_fit(rel: np.ndarray, ref: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    w_sum = float(np.sum(weights))
    w_rel = float(np.dot(weights, rel))
    w_ref = float(np.dot(weights, ref))
    w_rel_rel = float(np.dot(weights, rel * rel))
    w_rel_ref = float(np.dot(weights, rel * ref))
    denom = w_sum * w_rel_rel - w_rel * w_rel
    if abs(denom) <= 1e-8:
        raise ValueError("Degenerate weighted scale/shift fit.")
    scale = (w_sum * w_rel_ref - w_rel * w_ref) / denom
    shift = (w_rel_rel * w_ref - w_rel * w_rel_ref) / denom
    return float(scale), float(shift)


def _weighted_invdepth_fit(rel: np.ndarray, ref: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    inv_rel = np.reciprocal(np.clip(rel, 1e-6, None), dtype=np.float32)
    inv_ref = np.reciprocal(np.clip(ref, 1e-6, None), dtype=np.float32)
    return _weighted_scale_shift_fit(inv_rel, inv_ref, weights)


def _candidate_models(strategy: str) -> Tuple[AlignmentModel, ...]:
    strategy = strategy.lower()
    if strategy == "scale":
        return (AlignmentModel.SCALE,)
    if strategy == "scale_shift":
        return (AlignmentModel.SCALE_SHIFT,)
    if strategy in {"invdepth_scale_shift", "invdepth"}:
        return (AlignmentModel.INVDEPTH_SCALE_SHIFT,)
    return tuple(AlignmentModel)


@dataclass
class _EKFState:
    x: np.ndarray  # (scale, shift)
    P: np.ndarray  # covariance 2x2
    model: AlignmentModel


class MetricDepthAligner:
    """
    Per-camera Extended Kalman Filter that fuses weighted least-squares
    alignment with temporal smoothing.  The aligner can optionally look at
    TSDF residuals to inflate measurement covariances when the geometry is
    inconsistent (e.g. object motion).
    """

    def __init__(
        self,
        *,
        strategy: str = "scale_shift",
        max_samples: int = 20000,
        process_noise: Tuple[float, float] = (1e-4, 1e-4),
        measurement_floor: float = 1e-4,
        tsdf_influence: float = 4.0,
        rng_seed: int = 42,
    ) -> None:
        self.strategy = strategy
        self.max_samples = int(max_samples)
        self.measurement_floor = float(measurement_floor)
        self.tsdf_influence = float(tsdf_influence)
        self._states: Dict[str, _EKFState] = {}
        self._rng = np.random.default_rng(rng_seed)
        self._ray_cache: Dict[Tuple[int, int, float, float, float, float], np.ndarray] = {}
        self.Q = np.diag([process_noise[0], process_noise[1]]).astype(np.float32)

    # ------------------------------------------------------------------ helpers
    def set_state(self, camera_id: str, scale: float, shift: float, model: AlignmentModel = AlignmentModel.SCALE_SHIFT) -> None:
        x = np.array([scale, shift], dtype=np.float32)
        P = np.diag([0.05, 0.05]).astype(np.float32)
        self._states[camera_id] = _EKFState(x=x, P=P, model=model)

    def get_state(self, camera_id: str) -> Optional[Tuple[float, float]]:
        state = self._states.get(camera_id)
        if state is None:
            return None
        return float(state.x[0]), float(state.x[1])

    def _camera_rays(self, intrinsics: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
        cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
        key = (shape[0], shape[1], fx, fy, cx, cy)
        cached = self._ray_cache.get(key)
        if cached is not None:
            return cached
        h, w = shape
        u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        dirs = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1)
        norm = np.linalg.norm(dirs, axis=-1, keepdims=True)
        rays = dirs / np.clip(norm, 1e-6, None)
        self._ray_cache[key] = rays
        return rays

    @staticmethod
    def _compute_weights(reference: np.ndarray, mask: np.ndarray) -> np.ndarray:
        ref_safe = np.nan_to_num(reference, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        gy, gx = np.gradient(ref_safe)
        grad_mag = np.sqrt(gx * gx + gy * gy, dtype=np.float32)
        depth_scale = float(np.nanmedian(ref_safe[mask]) if np.any(mask) else 1.0)
        depth_scale = max(depth_scale, 1e-3)
        weights = 1.0 / (1.0 + grad_mag**2)
        weights *= 1.0 / (1.0 + (ref_safe / (depth_scale * 5.0)) ** 2)
        weights[~mask] = 0.0
        return weights.astype(np.float32)

    def _weighted_fit(
        self,
        rel: np.ndarray,
        ref: np.ndarray,
        weights: np.ndarray,
        models: Iterable[AlignmentModel],
    ) -> AlignmentResult:
        weights = weights.astype(np.float32)
        weights /= max(np.sum(weights), 1e-6)
        best: Optional[AlignmentResult] = None
        for model in models:
            try:
                if model is AlignmentModel.SCALE:
                    scale, shift = _weighted_scale_fit(rel, ref, weights)
                elif model is AlignmentModel.SCALE_SHIFT:
                    scale, shift = _weighted_scale_shift_fit(rel, ref, weights)
                elif model is AlignmentModel.INVDEPTH_SCALE_SHIFT:
                    scale, shift = _weighted_invdepth_fit(rel, ref, weights)
                else:  # pragma: no cover
                    continue
            except ValueError:
                continue
            metric = _apply_model(scale, shift, rel, model)
            residuals = ref - metric
            weighted_abs = np.dot(weights, np.abs(residuals))
            result = AlignmentResult(
                model=model,
                scale=float(scale),
                shift=float(shift),
                residual=float(weighted_abs),
                num_samples=rel.size,
            )
            if best is None or result.residual < best.residual:
                best = result
        if best is None:
            raise ValueError("Weighted alignment failed to converge.")
        return best

    # ---------------------------------------------------------------- alignment
    def align(
        self,
        camera_id: str,
        rel_depth: np.ndarray,
        reference_depth: np.ndarray,
        intrinsics: np.ndarray,
        pose_wc: np.ndarray,
        *,
        mask: Optional[np.ndarray] = None,
        tsdf_sampler: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> AlignmentResult:
        h, w = rel_depth.shape
        flat_rel = rel_depth.reshape(-1)
        flat_ref = reference_depth.reshape(-1)
        valid_mask = np.isfinite(flat_rel) & np.isfinite(flat_ref) & (flat_rel > 0) & (flat_ref > 0)
        if mask is not None:
            valid_mask &= mask.reshape(-1)
        valid_indices = np.nonzero(valid_mask)[0]
        if valid_indices.size < 64:
            raise ValueError("Insufficient overlapping pixels for alignment.")

        if valid_indices.size > self.max_samples:
            selected = self._rng.choice(valid_indices, size=self.max_samples, replace=False)
        else:
            selected = valid_indices

        rel_samples = flat_rel[selected].astype(np.float32)
        ref_samples = flat_ref[selected].astype(np.float32)
        per_pixel_mask = np.zeros_like(flat_rel, dtype=bool)
        per_pixel_mask[selected] = True
        weights_full = self._compute_weights(reference_depth, per_pixel_mask.reshape(h, w))
        weights_samples = weights_full.reshape(-1)[selected]
        if np.all(weights_samples <= 0):
            weights_samples = np.ones_like(rel_samples)

        result = self._weighted_fit(
            rel_samples,
            ref_samples,
            weights_samples,
            _candidate_models(self.strategy),
        )

        # Enforce positive scale
        scale = max(result.scale, 1e-6)
        shift = result.shift

        metric_samples = _apply_model(scale, shift, rel_samples, result.model)
        residuals = ref_samples - metric_samples
        weights_norm = weights_samples / max(float(np.sum(weights_samples)), 1e-6)
        rms = float(np.sqrt(np.dot(weights_norm, residuals**2)))
        residual_sigma = max(rms, self.measurement_floor)
        sigma_scale = residual_sigma / max(float(np.mean(rel_samples)), 1e-3)
        sigma_scale = max(sigma_scale, self.measurement_floor)
        sigma_shift = max(residual_sigma, self.measurement_floor)

        inflation = 1.0
        if tsdf_sampler is not None:
            rays = self._camera_rays(intrinsics, (h, w)).reshape(-1, 3)[selected]
            metric_depth = metric_samples
            points_c = rays * metric_depth[:, None]
            homog = np.concatenate([points_c, np.ones((points_c.shape[0], 1), dtype=np.float32)], axis=1)
            points_w = (pose_wc @ homog.T).T[:, :3]
            tsdf_vals = tsdf_sampler(points_w)
            if tsdf_vals.size > 0 and np.any(np.isfinite(tsdf_vals)):
                tsdf_residual = float(np.nanmedian(np.abs(tsdf_vals)))
                inflation = 1.0 + self.tsdf_influence * tsdf_residual**2

        R = np.diag([sigma_scale**2 * inflation, sigma_shift**2 * inflation]).astype(np.float32)

        state = self._states.get(camera_id)
        z = np.array([scale, shift], dtype=np.float32)
        if state is None:
            P0 = np.diag([sigma_scale**2, sigma_shift**2]).astype(np.float32)
            self._states[camera_id] = _EKFState(x=z, P=P0, model=result.model)
            filtered = z
        else:
            x_prior = state.x
            P_prior = state.P + self.Q
            S = P_prior + R
            try:
                K = P_prior @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                K = P_prior @ np.linalg.pinv(S)
            innovation = z - x_prior
            x_post = x_prior + K @ innovation
            P_post = (np.eye(2, dtype=np.float32) - K) @ P_prior
            state.x = x_post.astype(np.float32)
            state.P = P_post.astype(np.float32)
            state.model = result.model
            filtered = state.x

        filtered_scale = float(max(filtered[0], 1e-6))
        filtered_shift = float(filtered[1])
        return AlignmentResult(
            model=result.model,
            scale=filtered_scale,
            shift=filtered_shift,
            residual=residual_sigma,
            num_samples=result.num_samples,
        )
