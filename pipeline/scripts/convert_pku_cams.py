#!/usr/bin/env python3
"""
Convert PKU/MVSNet-style camera files (cam.txt) to this pipeline's JSON formats.

Input per camera file layout (example):

extrinsic
r11 r12 r13 t1
r21 r22 r23 t2
r31 r32 r33 t3
0 0 0 1

intrinsic
fx 0 cx
0 fy cy
0 0 1

<depth params line>  # ignored

Notes:
- The provided "extrinsic" is world-to-camera [R|t] (i.e., x_c = R x_w + t).
  The pipeline expects camera-to-world T_wc, so we output inv([R|t]).
- The intrinsic is K at the native image resolution; do not pre-scale.
- Output filenames: intrinsics.json and poses.json
  Compatible with onnx/pcd/io_utils.load_intrinsics/load_poses.

Usage:
  python scripts/convert_pku_cams.py \
    --cams-dir datasets/pku/1080_Kungfu_Fan_Single_m12/cams \
    --id-pattern "{stem}" \
    --out-dir datasets/pku/1080_Kungfu_Fan_Single_m12

By default, camera_id comes from each file stem. You can map to e.g. cam_{n}
with --id-pattern.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def parse_cam_txt(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    # Find 'extrinsic' and 'intrinsic' blocks
    try:
        e_idx = lines.index("extrinsic")
        i_idx = lines.index("intrinsic")
    except ValueError as e:
        raise RuntimeError(f"{path}: missing 'extrinsic' or 'intrinsic' sections") from e

    # Extrinsic block: next 4 lines are a 4x4 matrix (world-to-camera)
    E_rows = []
    for k in range(1, 5):
        parts = lines[e_idx + k].split()
        if len(parts) != 4:
            raise RuntimeError(f"{path}: invalid extrinsic row: {lines[e_idx + k]}")
        E_rows.append([float(x) for x in parts])
    E = np.asarray(E_rows, dtype=np.float64)
    if E.shape != (4, 4):
        raise RuntimeError(f"{path}: extrinsic must be 4x4, got {E.shape}")

    # Intrinsic block: next 3 lines are a 3x3 matrix
    K_rows = []
    for k in range(1, 4):
        parts = lines[i_idx + k].split()
        if len(parts) != 3:
            raise RuntimeError(f"{path}: invalid intrinsic row: {lines[i_idx + k]}")
        K_rows.append([float(x) for x in parts])
    K = np.asarray(K_rows, dtype=np.float64)
    if K.shape != (3, 3):
        raise RuntimeError(f"{path}: intrinsic must be 3x3, got {K.shape}")

    return K, E


def invert_extrinsic_to_T_wc(E: np.ndarray) -> np.ndarray:
    """
    Input E is world-to-camera: x_c = R x_w + t.
    Return T_wc (camera-to-world): x_w = R^T x_c - R^T t.
    """
    R = E[:3, :3]
    t = E[:3, 3]
    R_t = R.T
    t_wc = -R_t @ t
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_t
    T[:3, 3] = t_wc
    return T


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert PKU cam.txt to intrinsics/poses JSON")
    p.add_argument("--cams-dir", required=True, type=str, help="Directory containing *_cam.txt files")
    p.add_argument("--glob", default="*_cam.txt", type=str, help="Glob to match camera files")
    p.add_argument("--id-pattern", default="{stem}", type=str, help="Camera id pattern, e.g. 'cam_{n}' or '{stem}'")
    p.add_argument("--start-index", default=0, type=int, help="Starting index for {n} in id-pattern")
    p.add_argument("--out-dir", required=True, type=str, help="Output directory for intrinsics.json and poses.json")
    p.add_argument("--frame-id", default=0, type=int, help="Frame id to assign to static poses")
    return p


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    cams_dir = Path(args.cams_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(cams_dir, args.glob)
    files = sorted(cams_dir.glob(args.glob))
    if not files:
        raise SystemExit(f"No camera files match {args.glob} under {cams_dir}")

    intrinsics: Dict[str, list] = {}
    poses_entries = []

    n = int(args.start_index)
    for path in files:
        K, E = parse_cam_txt(path)
        T_wc = invert_extrinsic_to_T_wc(E)
        stem = path.stem  # e.g., 00000000_cam
        cam_id = args.id_pattern.format(stem=stem, n=n)
        n += 1

        intrinsics[cam_id] = np.asarray(K, dtype=np.float32).tolist()
        poses_entries.append({
            "camera": cam_id,
            "frame": int(args.frame_id),
            "matrix": np.asarray(T_wc, dtype=np.float32).tolist(),
        })

    intrinsics_json = {"intrinsics": intrinsics}
    poses_json = {"poses": poses_entries}

    with (out_dir / "intrinsics.json").open("w", encoding="utf-8") as f:
        json.dump(intrinsics_json, f, indent=2)
    with (out_dir / "poses.json").open("w", encoding="utf-8") as f:
        json.dump(poses_json, f, indent=2)

    print(f"Wrote {out_dir / 'intrinsics.json'} and {out_dir / 'poses.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
