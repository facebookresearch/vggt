"""TensorRT engine discovery helpers for PCD pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import logging

LOGGER = logging.getLogger(__name__)


class EnginePrecision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    INT8 = "int8"


AUTO_PRIORITY: Sequence[EnginePrecision] = (
    EnginePrecision.INT8,
    EnginePrecision.FP8,
    EnginePrecision.FP16,
    EnginePrecision.BF16,
    EnginePrecision.FP32,
)


def _list_engines(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*.engine") if p.is_file())


def discover_engines(engine_root: Path, precision_hint: Optional[str], base_name: Optional[str]) -> List[Path]:
    if not engine_root.exists():
        raise FileNotFoundError(engine_root)
    engines = _list_engines(engine_root)
    if precision_hint:
        engines = [p for p in engines if precision_hint.lower() in p.parent.name.lower() or precision_hint.lower() in p.name.lower()]
    if base_name:
        engines = [p for p in engines if base_name in p.name]
    return engines


def _precision_from_path(path: Path) -> EnginePrecision:
    for precision in EnginePrecision:
        if precision.value in path.parent.name.lower() or precision.value in path.name.lower():
            return precision
    return EnginePrecision.FP32


def select_engine(engines: Iterable[Path], preference: Optional[str]) -> Path:
    candidates = list(engines)
    if not candidates:
        raise RuntimeError("No TensorRT engines match the given filters.")
    if preference and preference.lower() != "auto":
        for candidate in candidates:
            if preference.lower() in candidate.parent.name.lower() or preference.lower() in candidate.name.lower():
                LOGGER.info("Selected engine %s (requested precision=%s)", candidate, preference)
                return candidate
        LOGGER.warning("Requested precision %s not available; falling back to automatic selection.", preference)
    ordered = sorted(candidates, key=lambda p: AUTO_PRIORITY.index(_precision_from_path(p)))
    LOGGER.info("Selected engine %s (auto precision)", ordered[0])
    return ordered[0]


@dataclass
class EngineInfo:
    path: Path
    precision: EnginePrecision
    size_bytes: int

    @classmethod
    def from_path(cls, path: Path) -> "EngineInfo":
        return cls(path=path, precision=_precision_from_path(path), size_bytes=path.stat().st_size)


def summarise_engines(engine_dir: Path) -> List[EngineInfo]:
    return [EngineInfo.from_path(p) for p in _list_engines(engine_dir)]

