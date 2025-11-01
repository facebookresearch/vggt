#!/usr/bin/env python3
"""
Legacy wrapper for :mod:`reconstruction.inference_pcd`.

This script is kept for historical reporting workflows but simply forwards to
the centralized pipeline entrypoint.  Prefer running

    python -m reconstruction.inference_pcd [...]
"""
from __future__ import annotations

import sys
from typing import Optional, Sequence

from .. import inference_pcd

_WARN_EMITTED = False


def _emit_warning() -> None:
    global _WARN_EMITTED
    if _WARN_EMITTED:
        return
    sys.stderr.write(
        "[warn] reconstruction.tools.pcd_inference is a legacy wrapper. Prefer: "
        "python -m reconstruction.inference_pcd ... (kept for reporting)\n"
    )
    sys.stderr.flush()
    _WARN_EMITTED = True


def main(argv: Optional[Sequence[str]] = None) -> int:
    _emit_warning()
    parser = inference_pcd.build_parser()
    args = parser.parse_args(argv)
    return inference_pcd.run_pipeline(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
