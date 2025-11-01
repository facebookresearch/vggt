"""
Compatibility shim for the relocated reconstruction pipeline.

The full implementation now lives under :mod:`reconstruction.inference_pcd`.
This module simply forwards calls so existing entry points (e.g.
``python -m onnx.inference_pcd``) continue to work while emitting a
deprecation warning.
"""
from __future__ import annotations

import argparse
import warnings
from typing import Optional, Sequence

from reconstruction.inference_pcd import build_parser, run_pipeline

__all__ = ["build_parser", "run_pipeline", "main"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    warnings.warn(
        "onnx.inference_pcd is deprecated; use reconstruction.inference_pcd instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_pipeline(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
