
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive per-device camera tester (V4L2 + OpenCV).

Keys:
  q   → next device
  p   → previous device
  s   → save a snapshot (./cam_snaps/cam<idx>_<timestamp>.jpg)
  Q/ESC → quit

Tries formats in this order (all with MJPG):
  1) 1920x1080 @ 60
  2) 1280x720  @ 60
  3) 1280x720  @ 30
  4) 640x360   @ 60

It overlays live FPS and the device path/title on the frame.
"""

import os
import glob
import cv2
import time
import math
from datetime import datetime

# ---------- helpers ----------

def list_video_devices(max_n=64):
    devs = []
    for n in range(max_n):
        p = f"/dev/video{n}"
        if os.path.exists(p):
            devs.append(p)
    return devs

def set_fourcc(cap, fourcc_str="MJPG"):
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)

def try_modes(cap, modes):
    """
    Try to apply modes in order; return the (w,h,fps) that actually sticks, or None.
    """
    for (w,h,fps) in modes:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS,          fps)
        # small settle
        time.sleep(0.05)
        # read a few frames to confirm
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        # verify dimensions
        h2, w2 = frame.shape[:2]
        # allow tiny mismatch (some cams deliver 640x368, etc.)
        if abs(w2 - w) <= 16 and abs(h2 - h) <= 16:
            # we got something—return the requested tuple
            return (w, h, fps)
    return None

def device_title(dev_path):
    # Try to read a friendly name from sysfs; fall back to path.
    # /sys/class/video4linux/videoX/name
    try:
        base = os.path.basename(dev_path)
        sys_name = f"/sys/class/video4linux/{base}/name"
        if os.path.exists(sys_name):
            with open(sys_name, "r", encoding="utf-8", errors="ignore") as f:
                name = f.read().strip()
            return f"{name} ({dev_path})"
    except Exception:
        pass
    return dev_path

def usb_root_hint(dev_path):
    # Try to show the USB host path (helps you see which root/hub it’s on).
    try:
        base = os.path.basename(dev_path)
        link = os.path.realpath(f"/sys/class/video4linux/{base}/device")
        # This path often contains ".../usbX/usbX-Y:1.Z/video4linux/..."
        # Extract the "...-Y" bit if present:
        parts = link.split("/")
        buses = [p for p in parts if "-" in p and ":" in p]
        if buses:
            return buses[-1].split(":")[0]  # e.g., "3-11.2"
    except Exception:
        pass
    return "unknown"

# ---------- main loop ----------

def run():
    devices = list_video_devices()
    if not devices:
        print("[ERR] No /dev/video* devices found.")
        return

    modes = [
        (1920,1080,60),
        (1280, 720,60),
        (1280, 720,30),
        ( 640, 360,60),
    ]

    os.makedirs("cam_snaps", exist_ok=True)

    idx = 0
    N = len(devices)
    print(f"[INFO] Found {N} device(s): {', '.join(devices)}")
    print("[INFO] Controls: q=next, p=prev, s=save snapshot, Q/ESC=quit")

    while 0 <= idx < N:
        dev = devices[idx]
        title = device_title(dev)
        usb = usb_root_hint(dev)

        print(f"\n[TEST] Opening {dev}  (root {usb})")
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)

        if not cap.isOpened():
            print(f"[FAIL] Could not open {dev}. (busy/permissions?)  Try: sudo usermod -aG video $USER && newgrp video")
            # move on
            idx += 1
            continue

        # Try MJPG first; many cams give best FPS with MJPG
        set_fourcc(cap, "MJPG")
        # keep only a couple of buffered frames to reduce latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        chosen = try_modes(cap, modes)
        if not chosen:
            print(f"[FAIL] {dev}: no tested mode worked (MJPG). Trying raw YUYV fallback…")
            set_fourcc(cap, "YUYV")
            chosen = try_modes(cap, modes)

        if not chosen:
            print(f"[FAIL] {dev}: could not grab a frame in any mode.")
            cap.release()
            idx += 1
            continue

        w, h, fps_req = chosen
        print(f"[OK] {dev}: streaming approximately {w}x{h} @ ~{fps_req} (requested)")

        win = f"[{idx+1}/{N}] {title} | USB {usb} | {w}x{h} req {fps_req}fps"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        # sized to about quarter of 4K so it’s not gigantic; you can resize manually
        cv2.resizeWindow(win, min(1280, w), min(720, h))

        # FPS measurement (EMA)
        ema = None
        alpha = 0.15
        last = time.perf_counter()
        good_frames = 0
        bad_reads = 0

        while True:
            ok, frame = cap.read()
            now = time.perf_counter()
            dt = now - last
            last = now

            if ok and frame is not None:
                good_frames += 1
                inst = (1.0 / dt) if dt > 1e-6 else 0.0
                ema = inst if ema is None else (alpha * inst + (1 - alpha) * ema)

                # overlay text
                txt1 = f"{title}"
                txt2 = f"USB {usb} | {w}x{h} | req {fps_req} fps | dec {ema:.1f} fps"
                cv2.putText(frame, txt1, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
                cv2.putText(frame, txt2, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)

                cv2.imshow(win, frame)
            else:
                bad_reads += 1
                # show a black frame with error
                black = 255 * (0 * (0)).__class__  # silly trick to avoid flake8 about "unused"
                err = (f"No signal / read failed (#{bad_reads}). "
                       f"Check cable/hub. Press q for next, p prev, Q/ESC quit.")
                img = (255 * (0)).__class__
                frame = (0,0,0)
                frame = (frame,)  # not necessary; just avoid None usage
                canvas = 255 * (0).__class__
                canvas = None
                # simpler: create a black image of requested size
                import numpy as np
                canvas = np.zeros((max(360,h), max(640,w), 3), dtype=np.uint8)
                cv2.putText(canvas, err, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)
                cv2.imshow(win, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'),):   # next
                idx += 1
                break
            if key in (ord('p'),):   # prev
                idx -= 1
                if idx < 0:
                    idx = 0
                break
            if key in (ord('Q'), 27):  # quit
                cap.release()
                cv2.destroyWindow(win)
                return
            if key in (ord('s'),):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                outp = os.path.join("cam_snaps", f"cam{idx}_{ts}.jpg")
                ok2, fr2 = cap.read()
                if ok2 and fr2 is not None:
                    cv2.imwrite(outp, fr2)
                    print(f"[SNAP] wrote {outp}")
                else:
                    print("[SNAP] failed to grab a frame")

        cap.release()
        cv2.destroyWindow(win)

    print("\n[DONE] Reached end of device list.")

if __name__ == "__main__":
    run()
