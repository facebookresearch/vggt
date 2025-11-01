#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Six-camera capture utility.

Features:
  - Auto-detects /dev/video nodes (or respects --devices order).
  - Displays a live 2x3 grid (configurable) similar to test_cameras.py.
  - On keypress ('s' by default) grabs a synchronized frame bundle and saves
    to the specified output directory, one subfolder per camera.
  - File names include a shared timestamp for cross-view alignment.
"""

import argparse
import glob
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


def fourcc(code: str) -> int:
    return cv2.VideoWriter_fourcc(*code)


def usb_path_for_video(dev: str) -> str:
    m = re.match(r"^/dev/video(\d+)$", dev)
    if not m:
        return "unknown"
    vid = m.group(1)
    base = f"/sys/class/video4linux/video{vid}"
    try:
        target = os.path.realpath(os.path.join(base, "device"))
        usb = re.search(r"(\d{4}:\d{2}:\d{2}\.\d-[\d\.]+)", target)
        return usb.group(1) if usb else os.path.basename(target)
    except Exception:
        return "unknown"


def sibling_node(path: str) -> Optional[str]:
    m = re.match(r"^/dev/video(\d+)$", path)
    if not m:
        return None
    i = int(m.group(1))
    sib = i ^ 1
    cand = f"/dev/video{sib}"
    return cand if os.path.exists(cand) else None


def try_open(dev: str, width: int, height: int, fps: int) -> Optional[cv2.VideoCapture]:
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    for pix in ("MJPG", "YUYV"):
        cap.set(cv2.CAP_PROP_FOURCC, fourcc(pix))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, float(fps))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        ok1, _ = cap.read()
        ok2, _ = cap.read()
        if ok1 or ok2:
            return cap
    cap.release()
    return None


def probe_with_fallback(dev: str, width: int, height: int, fps_candidates: List[int]) -> Tuple[Optional[str], Optional[cv2.VideoCapture], Optional[int]]:
    for candidate in (dev, sibling_node(dev)):
        if not candidate:
            continue
        for fps in fps_candidates:
            cap = try_open(candidate, width, height, fps)
            if cap is not None:
                return candidate, cap, fps
    return None, None, None


def auto_scan(limit: int, width: int, height: int, fps_candidates: List[int]) -> List[Tuple[str, cv2.VideoCapture, int]]:
    picked = []
    used = set()
    for i in range(0, 64):
        dev = f"/dev/video{i}"
        if not os.path.exists(dev):
            continue
        if (i ^ 1) in used:
            continue
        cand, cap, fps = probe_with_fallback(dev, width, height, fps_candidates)
        if cand and cap:
            picked.append((cand, cap, int(fps)))
            used.add(int(re.search(r"(\d+)$", cand).group(1)))
            print(f"[AUTO] Using {cand} @ {fps} fps  (USB {usb_path_for_video(cand)})")
        if len(picked) >= limit:
            break
    return picked


class CamThread:
    def __init__(self, dev: str, cap: cv2.VideoCapture, label: str):
        self.dev = dev
        self.cap = cap
        self.label = label
        self.lock = threading.Lock()
        self.frame = None
        self.stop = False
        self.fps_ema = None
        self.alpha = 0.15
        self.tlast = None
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def loop(self) -> None:
        while not self.stop:
            ret, frm = self.cap.read()
            now = time.time()
            if ret and frm is not None:
                with self.lock:
                    self.frame = frm
                if self.tlast is not None:
                    dt = now - self.tlast
                    if dt > 0:
                        fps = 1.0 / dt
                        self.fps_ema = fps if self.fps_ema is None else (1 - self.alpha) * self.fps_ema + self.alpha * fps
                self.tlast = now
            else:
                time.sleep(0.002)

    def read(self) -> Tuple[Optional[np.ndarray], float]:
        with self.lock:
            fr = None if self.frame is None else self.frame.copy()
        fps = 0.0 if self.fps_ema is None else float(self.fps_ema)
        return fr, fps

    def close(self) -> None:
        self.stop = True
        try:
            self.thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.cap.release()
        except Exception:
            pass


def put_text(img: np.ndarray, text: str, org=(16, 36), scale=1.0, thickness=2) -> None:
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def compose_grid(frames: List[Optional[np.ndarray]], rows: int, cols: int, win_w: int, win_h: int) -> np.ndarray:
    cell_w, cell_h = win_w // cols, win_h // rows
    tiles = []
    k = 0
    for r in range(rows):
        row_imgs = []
        for c in range(cols):
            if k < len(frames) and frames[k] is not None:
                img = cv2.resize(frames[k], (cell_w, cell_h), interpolation=cv2.INTER_AREA)
            else:
                img = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                put_text(img, "No signal", (cell_w // 6, cell_h // 2), 1.0, 2)
            row_imgs.append(img)
            k += 1
        tiles.append(np.hstack(row_imgs))
    return np.vstack(tiles)


def save_bundle(frames: List[np.ndarray], labels: List[str], out_dir: Path, prefix: str, ext: str = "jpg") -> None:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    for frame, label in zip(frames, labels):
        cam_dir = out_dir / label
        cam_dir.mkdir(parents=True, exist_ok=True)
        filename = cam_dir / f"{prefix}_{timestamp}.{ext}"
        cv2.imwrite(str(filename), frame)
    print(f"[SAVE] Captured bundle -> {timestamp} ({len(frames)} views)")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Multi-camera capture tool.")
    ap.add_argument("--devices", nargs="*", help="Device nodes (in grid order). Auto-detect if omitted.")
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--cap_w", type=int, default=1920)
    ap.add_argument("--cap_h", type=int, default=1080)
    ap.add_argument("--cap_fps", type=int, default=60)
    ap.add_argument("--fallback_fps", type=str, default="45,30")
    ap.add_argument("--win_w", type=int, default=3840)
    ap.add_argument("--win_h", type=int, default=2160)
    ap.add_argument("--title", default="Dataset Capture — press 's' to save bundle, 'q' to quit")
    ap.add_argument("--output", required=True, help="Destination directory for captured frames.")
    ap.add_argument("--prefix", default="capture", help="Filename prefix for saved frames.")
    ap.add_argument("--format", default="jpg", choices=["jpg", "png"], help="Image format for saved frames.")
    ap.add_argument("--save_key", default="s", help="Keyboard key (single char) to trigger capture.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    want = args.rows * args.cols
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    fps_candidates = [args.cap_fps] + [int(x) for x in args.fallback_fps.split(",") if x.strip().isdigit()]

    chosen = []
    if args.devices:
        if len(args.devices) != want:
            print(f"[WARN] Provided {len(args.devices)} devices but grid needs {want}. Will attempt auto-fill.")
        for dev in args.devices:
            cand, cap, fps = probe_with_fallback(dev, args.cap_w, args.cap_h, fps_candidates)
            if cand and cap:
                chosen.append((cand, cap, int(fps)))
                print(f"[PICK] {cand} @ {fps} fps  (USB {usb_path_for_video(cand)})")
        if len(chosen) < want:
            auto = auto_scan(want - len(chosen), args.cap_w, args.cap_h, fps_candidates)
            chosen.extend(auto)
        chosen = chosen[:want]
    else:
        chosen = auto_scan(want, args.cap_w, args.cap_h, fps_candidates)

    if not chosen:
        print("[ERROR] No working capture nodes found.")
        return

    threads: List[CamThread] = []
    labels: List[str] = []
    for dev, cap, fps in chosen:
        label = Path(dev).name  # e.g. video0
        label_dir = f"{label}"
        labels.append(label_dir)
        th = CamThread(dev, cap, f"{dev} @ {fps} ({usb_path_for_video(dev)})")
        th.start()
        threads.append(th)

    cv2.namedWindow(args.title, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(args.title, args.win_w, args.win_h)
    save_key = ord(args.save_key.lower())

    try:
        while True:
            frames, fpss = [], []
            for th in threads:
                fr, fps = th.read()
                frames.append(fr)
                fpss.append(fps)
            for idx, fr in enumerate(frames):
                if fr is None:
                    continue
                put_text(fr, f"{labels[idx]} | {fpss[idx]:5.1f} FPS", (16, 40), 1.0, 2)
            grid = compose_grid(frames, args.rows, args.cols, args.win_w, args.win_h)
            cv2.imshow(args.title, grid)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # q or ESC
                break
            if key == save_key:
                bundle_frames = [fr for fr in frames if fr is not None]
                if len(bundle_frames) != len(labels):
                    print("[WARN] Some cameras missing frame; skipping capture.")
                else:
                    save_bundle(bundle_frames, labels, out_dir, args.prefix, args.format)
    finally:
        for th in threads:
            th.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
