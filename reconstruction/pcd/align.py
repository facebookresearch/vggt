"""
Depth alignment utilities.

This module implements several lightweight models to convert relative depth
predictions into metric depths by aligning them to a reference map.  The
implementation favours robustness (trimmed statistics and median residuals)
over chasing exact optimality which keeps the runtime small enough for live
fusion.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional, Tuple

import numpy as np


class AlignmentModel(str, Enum):
    """Supported alignment models."""

    SCALE = "scale"
    SCALE_SHIFT = "scale_shift"
    INVDEPTH_SCALE_SHIFT = "invdepth_scale_shift"


@dataclass
class AlignmentResult:
    """Result of aligning relative depth to a metric reference."""

    model: AlignmentModel
    scale: float
    shift: float
    residual: float
    num_samples: int

    def apply(self, depth: np.ndarray) -> np.ndarray:
        """
        Apply the fitted model to ``depth``.
        """
        if self.model is AlignmentModel.INVDEPTH_SCALE_SHIFT:
            inv = np.reciprocal(np.clip(depth, 1e-6, None), dtype=np.float32)
            aligned = np.reciprocal(np.clip(self.scale * inv + self.shift, 1e-6, None), dtype=np.float32)
            return aligned
        return self.scale * depth + self.shift


def _prepare_data(
    rel_depth: np.ndarray,
    metric_depth: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    rel = rel_depth.astype(np.float32).reshape(-1)
    met = metric_depth.astype(np.float32).reshape(-1)
    if mask is not None:
        valid = mask.astype(bool).reshape(-1)
    else:
        valid = np.ones_like(rel, dtype=bool)
    valid &= np.isfinite(rel) & np.isfinite(met)
    rel, met = rel[valid], met[valid]
    if rel.size == 0:
        raise ValueError("No valid samples for alignment.")
    # Trim the extreme 5% tails for robustness.
    lower = np.percentile(rel, 5)
    upper = np.percentile(rel, 95)
    clip_mask = (rel >= lower) & (rel <= upper)
    rel, met = rel[clip_mask], met[clip_mask]
    return rel, met


def _median_abs(residuals: np.ndarray) -> float:
    return float(np.median(np.abs(residuals)))


def _fit_scale(rel: np.ndarray, met: np.ndarray) -> AlignmentResult:
    scale = float(np.dot(rel, met) / max(np.dot(rel, rel), 1e-6))
    residual = _median_abs(met - scale * rel)
    return AlignmentResult(
        model=AlignmentModel.SCALE,
        scale=scale,
        shift=0.0,
        residual=residual,
        num_samples=rel.size,
    )


def _fit_scale_shift(rel: np.ndarray, met: np.ndarray) -> AlignmentResult:
    A = np.stack([rel, np.ones_like(rel)], axis=1)
    solution, *_ = np.linalg.lstsq(A, met, rcond=None)
    scale, shift = map(float, solution)
    residual = _median_abs(met - (scale * rel + shift))
    return AlignmentResult(
        model=AlignmentModel.SCALE_SHIFT,
        scale=scale,
        shift=shift,
        residual=residual,
        num_samples=rel.size,
    )


def _fit_invdepth_scale_shift(rel: np.ndarray, met: np.ndarray) -> AlignmentResult:
    inv_rel = np.reciprocal(np.clip(rel, 1e-6, None), dtype=np.float32)
    inv_met = np.reciprocal(np.clip(met, 1e-6, None), dtype=np.float32)
    A = np.stack([inv_rel, np.ones_like(inv_rel)], axis=1)
    solution, *_ = np.linalg.lstsq(A, inv_met, rcond=None)
    scale, shift = map(float, solution)
    residual = _median_abs(inv_met - (scale * inv_rel + shift))
    return AlignmentResult(
        model=AlignmentModel.INVDEPTH_SCALE_SHIFT,
        scale=scale,
        shift=shift,
        residual=residual,
        num_samples=rel.size,
    )


def fit_models(
    rel_depth: np.ndarray,
    metric_depth: np.ndarray,
    models: Iterable[AlignmentModel],
    mask: Optional[np.ndarray] = None,
) -> Tuple[AlignmentResult, ...]:
    """
    Fit the requested alignment ``models`` and return their results.
    """
    rel, met = _prepare_data(rel_depth, metric_depth, mask)
    results = []
    for model in models:
        if model is AlignmentModel.SCALE:
            results.append(_fit_scale(rel, met))
        elif model is AlignmentModel.SCALE_SHIFT:
            results.append(_fit_scale_shift(rel, met))
        elif model is AlignmentModel.INVDEPTH_SCALE_SHIFT:
            results.append(_fit_invdepth_scale_shift(rel, met))
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unsupported alignment model: {model}")
    return tuple(results)


def auto_select_model(
    rel_depth: np.ndarray,
    metric_depth: np.ndarray,
    mask: Optional[np.ndarray],
    strategy: str = "auto",
) -> AlignmentResult:
    """
    Pick the best alignment model based on residuals.
    """
    strategy = strategy.lower()
    if strategy == "scale":
        models = (AlignmentModel.SCALE,)
    elif strategy == "scale_shift":
        models = (AlignmentModel.SCALE_SHIFT,)
    elif strategy in {"invdepth_scale_shift", "invdepth"}:
        models = (AlignmentModel.INVDEPTH_SCALE_SHIFT,)
    else:
        models = tuple(AlignmentModel)
    results = fit_models(rel_depth, metric_depth, models, mask)
    best = min(results, key=lambda r: r.residual)
    return best


class ScaleShiftEMA:
    """
    Exponential moving average smoother for per-camera scale/shift estimates.
    """

    def __init__(self, alpha: float) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be within [0, 1]")
        self.alpha = alpha
        self._state: dict[str, Tuple[float, float]] = {}

    def update(self, camera_id: str, result: AlignmentResult) -> AlignmentResult:
        """
        Smooth ``result`` for ``camera_id`` and return the smoothed estimate.
        """
        if self.alpha == 0.0:
            return result
        prev = self._state.get(camera_id)
        if prev is None:
            self._state[camera_id] = (result.scale, result.shift)
            return result
        new_scale = self.alpha * result.scale + (1.0 - self.alpha) * prev[0]
        new_shift = self.alpha * result.shift + (1.0 - self.alpha) * prev[1]
        self._state[camera_id] = (new_scale, new_shift)
        return AlignmentResult(
            model=result.model,
            scale=new_scale,
            shift=new_shift,
            residual=result.residual,
            num_samples=result.num_samples,
        )

    def set_state(self, camera_id: str, scale: float, shift: float) -> None:
        """Initialise the smoother with a cached scale/shift pair."""
        self._state[camera_id] = (scale, shift)

    def get_state(self, camera_id: str) -> Optional[Tuple[float, float]]:
        """Return the current state for ``camera_id`` if available."""
        return self._state.get(camera_id)

