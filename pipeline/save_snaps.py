#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grab exactly ONE frame from each connected /dev/video* camera, sequentially.

Features:
- Enumerates /dev/video0..63
- Sequential, low-burst USB usage
- Tries MJPG 1280x720 first (fast), then falls back to 1080p, then YUYV, then "whatever works"
- Warm-up frames + per-device timeout + retries
- Writes JPEGs and a manifest.json with device metadata, USB path, chosen mode, and timings

Usage:
  python3 grab_all_cams_once.py --out runs/take_001
  python3 grab_all_cams_once.py --out runs/take_002 --devs /dev/video0 /dev/video2 /dev/video4
"""

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ---------- Helpers ----------

def list_video_devices(max_nodes: int = 12) -> List[str]:
    devs = []
    for i in range(0,max_nodes,2):
        p = f"/dev/video{i}"
        if os.path.exists(p):
            devs.append(p)
    return devs

def read_sysfs_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except Exception:
        return None

def get_device_name(dev_path: str) -> str:
    base = os.path.basename(dev_path)
    sys_name = f"/sys/class/video4linux/{base}/name"
    name = read_sysfs_text(sys_name)
    return name if name else dev_path

def get_usb_path_hint(dev_path: str) -> str:
    """
    Returns something like '3-11.2' (USB bus-port path) if available, else 'unknown'.
    """
    try:
        base = os.path.basename(dev_path)
        link = os.path.realpath(f"/sys/class/video4linux/{base}/device")
        parts = link.split("/")
        # Look for segments like '3-11.2:1.0'
        candidates = [p for p in parts if "-" in p and ":" in p]
        if candidates:
            return candidates[-1].split(":")[0]
    except Exception:
        pass
    return "unknown"

def set_fourcc(cap, code: str):
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*code))

def try_config(cap, width: int, height: int, fps: int, settle: float = 0.03) -> bool:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS,          fps)
    time.sleep(settle)
    ok, frame = cap.read()
    if not ok or frame is None:
        return False
    h2, w2 = frame.shape[:2]
    # Accept minor deviations (some cams report 640x368, etc.)
    if abs(w2 - width) <= 16 and abs(h2 - height) <= 16:
        return True
    # If dims differ but we still got a frame, accept it as a last resort
    return True

def grab_one_frame(cap, warmup: int, timeout_s: float) -> Optional[np.ndarray]:
    t0 = time.time()
    # warmup (discard a few frames to settle exposure/auto focus)
    for _ in range(max(0, warmup)):
        ok, _ = cap.read()
        if not ok:
            break
    # timed loop to obtain one good frame
    while time.time() - t0 < timeout_s:
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame
    return None

# ---------- Core capture ----------

def capture_from_device(
    dev: str,
    out_dir: str,
    jpeg_quality: int = 85,
    warmup_frames: int = 3,
    timeout_s: float = 1.2,
    retries: int = 2,
    verbose: bool = True,
) -> Tuple[bool, Optional[str], dict]:
    """
    Attempts to open and capture one frame from 'dev'.
    Returns (success, saved_path, info_dict)
    """
    info = {
        "device": dev,
        "name": get_device_name(dev),
        "usb_path": get_usb_path_hint(dev),
        "attempts": [],
        "chosen": None,
        "error": None,
        "ms_open": None,
        "ms_total": None,
    }
    t_all0 = time.time()

    # Open the device
    t0 = time.time()
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        info["error"] = "open_failed"
        info["ms_open"] = int((time.time() - t0) * 1000)
        return (False, None, info)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    info["ms_open"] = int((time.time() - t0) * 1000)

    # Try prioritized configurations
    # Prefer MJPG @ 1280x720 (fast), then MJPG @ 1920x1080, then YUYV @ 1280x720
    configs = [
        ("MJPG", 1920, 1080, 60),
    ]

    chosen = None
    for fourcc, w, h, fps in configs:
        try:
            set_fourcc(cap, fourcc)
            ok = try_config(cap, w, h, fps)
            info["attempts"].append({"fourcc": fourcc, "w": w, "h": h, "fps": fps, "ok": bool(ok)})
            if ok:
                chosen = (fourcc, w, h, fps)
                break
        except Exception as e:
            info["attempts"].append({"fourcc": fourcc, "w": w, "h": h, "fps": fps, "ok": False, "exc": str(e)})

    # As a last resort, try *no* explicit config (driver default)
    if chosen is None:
        info["attempts"].append({"fourcc": "AUTO", "w": -1, "h": -1, "fps": -1, "ok": False})
        # give it one more shot to read anything at all
        pass

    # Try to read with retries
    frame = None
    for r in range(retries + 1):
        frame = grab_one_frame(cap, warmup_frames if r == 0 else max(1, warmup_frames // 2), timeout_s)
        if frame is not None:
            break
        # small delay before retry
        time.sleep(0.05)

    if frame is None:
        info["error"] = "timeout_no_frame"
        cap.release()
        info["ms_total"] = int((time.time() - t_all0) * 1000)
        return (False, None, info)

    # Successful capture — write JPEG
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base = os.path.basename(dev)
    fname = f"{base}_{ts}.jpg"
    save_path = os.path.join(out_dir, fname)
    ok = cv2.imwrite(save_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    if not ok:
        info["error"] = "imwrite_failed"
        cap.release()
        info["ms_total"] = int((time.time() - t_all0) * 1000)
        return (False, None, info)

    if chosen:
        info["chosen"] = {"fourcc": chosen[0], "w": chosen[1], "h": chosen[2], "fps": chosen[3]}
    else:
        # Unknown mode (driver default)
        h2, w2 = frame.shape[:2]
        info["chosen"] = {"fourcc": "UNKNOWN", "w": w2, "h": h2, "fps": -1}

    cap.release()
    info["ms_total"] = int((time.time() - t_all0) * 1000)
    return (True, save_path, info)

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Grab one frame from each /dev/video* sequentially (robust).")
    ap.add_argument("--out", required=True, help="Output directory (frames + manifest.json)")
    ap.add_argument("--devs", nargs="*", default=None, help="Optional explicit device list (e.g. /dev/video0 /dev/video2 ...)")
    ap.add_argument("--max-nodes", type=int, default=64, help="Scan up to /dev/video{N-1} (default 64)")
    ap.add_argument("--jpegq", type=int, default=100, help="JPEG quality (0-100)")
    ap.add_argument("--warmup", type=int, default=2, help="Warm-up frames to discard before capture")
    ap.add_argument("--timeout", type=float, default=0.2, help="Per-device read timeout (seconds)")
    ap.add_argument("--retries", type=int, default=1, help="Per-device retries if no frame")
    ap.add_argument("--verbose", action="store_true", help="Verbose logs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.json")

    if args.devs:
        devices = args.devs
    else:
        devices = list_video_devices(args.max_nodes)

    if not devices:
        print("[ERR] No /dev/video* devices found.", file=sys.stderr)
        sys.exit(2)

    summary = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "out_dir": os.path.abspath(args.out),
        "devices": [],
        "global": {
            "jpeg_quality": args.jpegq,
            "warmup_frames": args.warmup,
            "timeout_s": args.timeout,
            "retries": args.retries,
        },
    }

    print(f"[INFO] Found {len(devices)} device(s): {' '.join(devices)}")
    ok_count = 0
    fail_count = 0

    for dev in devices:
        print(f"[...] Capturing from {dev} …")
        os.sync()
        try:
            success, saved, info = capture_from_device(
                dev=dev,
                out_dir=args.out,
                jpeg_quality=args.jpegq,
                warmup_frames=args.warmup,
                timeout_s=args.timeout,
                retries=args.retries,
                verbose=args.verbose,
            )
        except Exception as e:
            info = {
                "device": dev,
                "name": get_device_name(dev),
                "usb_path": get_usb_path_hint(dev),
                "attempts": [],
                "chosen": None,
                "error": f"exception: {e!r}",
                "ms_open": None,
                "ms_total": None,
            }
            success = False
            saved = None

        info["saved_path"] = saved
        summary["devices"].append(info)

        if success:
            ok_count += 1
            chosen = info.get("chosen") or {}
            print(f"[OK] {dev} -> {os.path.basename(saved)} "
                  f"({chosen.get('fourcc','?')} {chosen.get('w','?')}x{chosen.get('h','?')} "
                  f"fps={chosen.get('fps','?')}) in {info.get('ms_total','?')} ms")
        else:
            fail_count += 1
            print(f"[SKIP] {dev} ({info.get('error')})")

        # Light sync point between devices to avoid hammering USB
        time.sleep(0.03)

    summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
    summary["stats"] = {"ok": ok_count, "failed": fail_count, "total": len(devices)}

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[SUMMARY] ok={ok_count} failed={fail_count} total={len(devices)}")
    print(f"[WRITE]   {manifest_path}")

if __name__ == "__main__":
    main()
