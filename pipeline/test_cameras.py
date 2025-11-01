#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Six-camera 2x3 grid viewer with:
- USB topology printout per device
- Sibling auto-pick (/dev/videoN <-> N^1)
- Adaptive FPS fallback (60 -> 45 -> 30) to fill the grid

Press 'q' to quit.
"""
import argparse, os, re, time, threading, subprocess, shlex, glob
import cv2, numpy as np

# ---------- helpers ----------
def fourcc(code: str) -> int:
    return cv2.VideoWriter_fourcc(*code)

def run(cmd: str) -> str:
    try:
        r = subprocess.run(shlex.split(cmd), capture_output=True, text=True, timeout=1.5)
        return r.stdout.strip()
    except Exception:
        return ""

def usb_path_for_video(dev: str) -> str:
    # /sys/class/video4linux/videoX/device -> ../../../usb.../X-Y:1.0/video4linux/videoX
    m = re.match(r"^/dev/video(\d+)$", dev)
    if not m: return "unknown"
    vid = m.group(1)
    base = f"/sys/class/video4linux/video{vid}"
    try:
        target = os.path.realpath(os.path.join(base, "device"))
        # extract USB bus-port path like "0000:80:14.0-11.1"
        usb = re.search(r"(\d{4}:\d{2}:\d{2}\.\d-[\d\.]+)", target)
        return usb.group(1) if usb else os.path.basename(target)
    except Exception:
        return "unknown"

def list_formats(dev: str) -> str:
    return run(f"v4l2-ctl -d {dev} --list-formats-ext")

def sibling_node(path: str) -> str | None:
    m = re.match(r"^/dev/video(\d+)$", path)
    if not m: return None
    i = int(m.group(1))
    sib = i ^ 1
    cand = f"/dev/video{sib}"
    return cand if os.path.exists(cand) else None

def try_open(dev: str, width=1920, height=1080, fps=60):
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    for pix in ("MJPG", "YUYV"):
        cap.set(cv2.CAP_PROP_FOURCC, fourcc(pix))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, float(fps))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        # Prime a couple frames
        ok1, _ = cap.read()
        ok2, _ = cap.read()
        if ok1 or ok2:
            return cap
    cap.release()
    return None

def probe_with_fallback(dev: str, width, height, fps_candidates):
    """Try dev, then its sibling. For each, try fps list in order; return (picked_dev, cap, fps_used)."""
    for candidate in (dev, sibling_node(dev)):
        if not candidate: 
            continue
        for f in fps_candidates:
            cap = try_open(candidate, width, height, f)
            if cap is not None:
                return candidate, cap, f
    return None, None, None

def auto_scan(limit, width, height, fps_candidates):
    picked = []
    used = set()
    for i in range(0, 64):
        dev = f"/dev/video{i}"
        if not os.path.exists(dev):
            continue
        # avoid selecting both nodes of same cam
        if (i ^ 1) in used:
            continue
        cand, cap, used_fps = probe_with_fallback(dev, width, height, fps_candidates)
        if cand and cap:
            picked.append((cand, cap, used_fps))
            used.add(int(re.search(r"(\d+)$", cand).group(1)))
            print(f"[AUTO] Using {cand} @ {used_fps} fps  (USB {usb_path_for_video(cand)})")
        if len(picked) >= limit:
            break
    return picked

# ---------- capture thread ----------
class CamThread:
    def __init__(self, dev, cap, label):
        self.dev, self.cap, self.label = dev, cap, label
        self.lock = threading.Lock()
        self.frame = None
        self.stop = False
        self.fps_ema = None
        self.alpha = 0.15
        self.tlast = None
        self.th = threading.Thread(target=self.loop, daemon=True)

    def start(self): self.th.start()

    def loop(self):
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
                        self.fps_ema = fps if self.fps_ema is None else (1-self.alpha)*self.fps_ema + self.alpha*fps
                self.tlast = now
            else:
                time.sleep(0.001)

    def read(self):
        with self.lock:
            f = None if self.frame is None else self.frame.copy()
        return f, (0.0 if self.fps_ema is None else self.fps_ema)

    def close(self):
        self.stop = True
        try: self.th.join(timeout=1.0)
        except: pass
        try: self.cap.release()
        except: pass

# ---------- UI ----------
def put_text(img, text, org=(18,38), scale=1.1, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), thickness, cv2.LINE_AA)

def compose_grid(frames, rows, cols, W, H):
    cw, ch = W // cols, H // rows
    tiles = []
    k = 0
    for r in range(rows):
        row = []
        for c in range(cols):
            if k < len(frames) and frames[k] is not None:
                img = cv2.resize(frames[k], (cw, ch), interpolation=cv2.INTER_AREA)
            else:
                img = np.zeros((ch, cw, 3), dtype=np.uint8)
                put_text(img, "No signal", (cw//3, ch//2), 1.0, 2)
            row.append(img); k += 1
        tiles.append(np.hstack(row))
    return np.vstack(tiles)

# ---------- main ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", nargs="*", help="Six device nodes in grid order (omit to auto-pick)")
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--cap_w", type=int, default=1920)
    ap.add_argument("--cap_h", type=int, default=1080)
    ap.add_argument("--cap_fps", type=int, default=60, help="Target FPS per cam")
    ap.add_argument("--fallback_fps", type=str, default="45,30", help="Comma list used if not all cams can open")
    ap.add_argument("--win_w", type=int, default=3840)
    ap.add_argument("--win_h", type=int, default=2160)
    ap.add_argument("--title", default="6-cam grid — press 'q' to quit")
    return ap.parse_args()

def main():
    args = parse_args()
    want = args.rows * args.cols

    # Show USB topology for all present nodes so you can re-plug intelligently
    print("\n[TOPOLOGY] Present /dev/video* and USB paths:")
    for path in sorted(glob.glob("/dev/video*"), key=lambda p:int(re.search(r"(\d+)$", p).group(1))):
        print(f"  {path:>12}  ->  USB {usb_path_for_video(path)}")
    print()

    fps_candidates = [args.cap_fps] + [int(x) for x in args.fallback_fps.split(",") if x.strip().isdigit()]

    chosen = []
    if args.devices:
        if len(args.devices) != want:
            print(f"[WARN] Provided {len(args.devices)} devices but grid needs {want}. I’ll try to fill.")
        for dev in args.devices:
            cand, cap, used_fps = probe_with_fallback(dev, args.cap_w, args.cap_h, fps_candidates)
            if cand and cap:
                chosen.append((cand, cap, used_fps))
                print(f"[PICK] {cand} @ {used_fps} fps  (USB {usb_path_for_video(cand)})")
        if len(chosen) < want:
            print("[INFO] Not enough working devices from your list; auto-scanning to fill the rest…")
            auto = auto_scan(limit=want-len(chosen), width=args.cap_w, height=args.cap_h, fps_candidates=fps_candidates)
            chosen.extend(auto)
        chosen = chosen[:want]
    else:
        chosen = auto_scan(limit=want, width=args.cap_w, height=args.cap_h, fps_candidates=fps_candidates)

    if len(chosen) == 0:
        print("[ERROR] No working capture nodes found.")
        return

    # spin up threads
    threads, labels = [], []
    for dev, cap, used_fps in chosen:
        label = f"{dev} @ {used_fps}  ({os.path.basename(usb_path_for_video(dev))})"
        th = CamThread(dev, cap, label); th.start()
        threads.append(th); labels.append(label)

    cv2.namedWindow(args.title, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
    cv2.resizeWindow(args.title, args.win_w, args.win_h)

    try:
        while True:
            frames, fpss = [], []
            for th in threads:
                f, fps = th.read(); frames.append(f); fpss.append(fps)
            for i,f in enumerate(frames):
                if f is None: continue
                put_text(f, f"{labels[i]}  |  {fpss[i]:5.1f} FPS")
            grid = compose_grid(frames, args.rows, args.cols, args.win_w, args.win_h)
            cv2.imshow(args.title, grid)
            k = cv2.waitKey(1)
            if k == ord('q') or k == 27:
                break
    finally:
        for th in threads: th.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
