#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual-camera DepthAnything v2 (TensorRT 10+) → live **ORTHOGRAPHIC** point clouds + compact RGB/Depth monitor.

Designed for **FPS**:
- Two TRT execution contexts + two CUDA streams (LEFT/RIGHT in parallel)
- Optional **CuPy** ortho projector + **stride** decimation
- Outlier removal / normals **off by default** (you can run them sparsely)
- Fitting windows: Open3D sizes configurable; monitor scaled

Keys: [q]=quit, [s]=snapshot (PNG + two PLYs)
"""

import argparse, time, threading, queue, ctypes, sys, os, shutil, subprocess, math
from typing import Tuple, Optional, List, Dict
from pathlib import Path
import numpy as np, cv2

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reconstruction.pcd.depth_anything import DepthAnythingPool

# -------------------------- CuPy & Open3D ------------------------------------
try:
    import cupy as cp
    _HAS_CUPY = True
except Exception:
    cp = None
    _HAS_CUPY = False

try:
    import open3d as o3d
    _HAS_O3D = True
except Exception:
    o3d = None
    _HAS_O3D = False

# -------------------------- Preproc / Viz ------------------------------------
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD  = np.array([0.229, 0.224, 0.225], np.float32)

class RunningScaler:
    def __init__(self, p_low=2.0, p_high=98.0, momentum=0.90):
        self.pl = float(p_low); self.ph = float(p_high)
        self.m = float(momentum)
        self.lo = None; self.hi = None
    def update(self, depth_valid: np.ndarray):
        v = depth_valid
        if v.size < 200:
            return self.lo, self.hi
        if v.size > 20000:
            idx = np.random.choice(v.size, 20000, replace=False)
            v = v[idx]
        lo, hi = np.percentile(v, [self.pl, self.ph])
        if self.lo is None or self.hi is None:
            self.lo, self.hi = float(lo), float(hi)
        else:
            self.lo = self.m*self.lo + (1-self.m)*float(lo)
            self.hi = self.m*self.hi + (1-self.m)*float(hi)
        return self.lo, self.hi

def letterbox(img: np.ndarray, side: int) -> Tuple[np.ndarray, Tuple[int,int,int,int], Tuple[int,int]]:
    h, w = img.shape[:2]
    s = min(side/h, side/w)
    nh, nw = int(round(h*s)), int(round(w*s))
    img_res = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
    pt = (side-nh)//2; pb = side-nh-pt; pl = (side-nw)//2; pr = side-nw-pl
    img_pad = cv2.copyMakeBorder(img_res, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=(0,0,0))
    return img_pad, (pt,pb,pl,pr), (h,w)

def preprocess(bgr: np.ndarray, side: int):
    sq, pads, hw = letterbox(bgr, side)
    rgb = cv2.cvtColor(sq, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    rgb = (rgb - MEAN)/STD
    chw = rgb.transpose(2,0,1)[None].astype(np.float32, copy=False)
    return np.ascontiguousarray(chw), pads, hw

def _squeeze_hw(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    while a.ndim > 2:
        a = np.squeeze(a, axis=0)
    return a

def resize_depth_to_original(depth_sq: np.ndarray, pads, out_hw: Tuple[int,int]) -> np.ndarray:
    d = _squeeze_hw(depth_sq).astype(np.float32)
    S0, S1 = d.shape[:2]
    pt,pb,pl,pr = pads
    r0, r1 = max(0, pt), max(0, pb)
    c0, c1 = max(0, pl), max(0, pr)
    core = d[r0:S0-r1 if S0-r1>r0 else S0, c0:S1-c1 if S1-c1>c0 else S1]
    if core.size == 0:
        return np.zeros((out_hw[0], out_hw[1]), np.float32)
    return cv2.resize(core, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR)

def colorize_scaled(depth_sq: np.ndarray, pads, out_hw: Tuple[int,int], lo: Optional[float], hi: Optional[float]) -> np.ndarray:
    d = _squeeze_hw(depth_sq).astype(np.float32)
    S0, S1 = d.shape[:2]
    pt,pb,pl,pr = pads
    r0, r1 = max(0, pt), max(0, pb)
    c0, c1 = max(0, pl), max(0, pr)
    core = d[r0:S0-r1 if S0-r1>r0 else S0, c0:S1-c1 if S1-c1>c0 else S1]
    fin = np.isfinite(core)
    if not np.any(fin) or lo is None or hi is None or (hi - lo) < 1e-6:
        return np.zeros((out_hw[0], out_hw[1], 3), np.uint8)
    m = np.clip((core - lo) / (hi - lo), 0, 1)
    u8 = (m * 255).astype(np.uint8)
    u8 = cv2.resize(u8, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_LINEAR)
    return cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)

def depth_to_u8(depth: np.ndarray) -> np.ndarray:
    v = depth[np.isfinite(depth)]
    if v.size == 0:
        return np.zeros_like(depth, np.uint8)
    dmax = float(np.max(v))
    if dmax <= 1e-12:
        return np.zeros_like(depth, np.uint8)
    return (np.clip(depth / dmax, 0, 1) * 255.0).astype(np.uint8)


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".avi", ".mpg", ".mpeg", ".mp4v")


def _collect_images(folder: Path) -> List[Path]:
    files: List[Path] = []
    for ext in IMAGE_EXTS:
        files.extend(sorted(folder.glob(f"*{ext}")))
    return files


def _collect_videos(folder: Path) -> List[Path]:
    files: List[Path] = []
    for ext in VIDEO_EXTS:
        files.extend(sorted(folder.glob(f"*{ext}")))
    return files


def _colorize_depth(depth: np.ndarray) -> np.ndarray:
    finite = np.isfinite(depth) & (depth > 0)
    if not np.any(finite):
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    lo, hi = np.percentile(depth[finite], [2.0, 98.0])
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    u8 = (norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)


def run_offline_grid(args) -> None:
    folder = Path(args.offline_dir).expanduser().resolve()
    if not folder.exists():
        raise FileNotFoundError(f"Input directory not found: {folder}")

    image_paths = _collect_images(folder)
    if not image_paths:
        video_paths = _collect_videos(folder)
        if video_paths:
            print(f"[info] Detected video files under {folder}; switching to video stream mode.")
            setattr(args, "offline_video_dir", str(folder))
            run_video_stream(args)
            return
        raise RuntimeError(f"No image files found under {folder}")

    if args.grid_limit and args.grid_limit > 0:
        image_paths = image_paths[: int(args.grid_limit)]

    images: Dict[str, np.ndarray] = {}
    timestamps: Dict[str, float] = {}
    for idx, path in enumerate(image_paths):
        capture_ts = path.stat().st_mtime
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[warn] Skipping unreadable image: {path}")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        key = f"{idx:03d}_{path.stem}"
        images[key] = rgb
        timestamps[key] = capture_ts
    if not images:
        raise RuntimeError("No valid images to process.")

    engine_path = Path(args.engine).expanduser().resolve()
    with DepthAnythingPool(
        engine_path.parent,
        precision="auto",
        num_workers=max(1, int(args.engine_workers)),
        explicit_engine=engine_path,
    ) as pool:
        start = time.perf_counter()
        depths = pool.infer_batch(images)
        elapsed = time.perf_counter() - start

    ordered_keys = list(images.keys())
    tile_h, tile_w = next(iter(depths.values())).shape
    if args.grid_cols and args.grid_cols > 0:
        cols = min(int(args.grid_cols), len(ordered_keys))
    else:
        cols = len(ordered_keys)
    cols = max(1, cols)
    rows = math.ceil(len(ordered_keys) / cols)
    grid = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

    for idx, key in enumerate(ordered_keys):
        depth = depths.get(key)
        if depth is None:
            continue
        colored = _colorize_depth(depth.astype(np.float32))
        if colored.shape[0] != tile_h or colored.shape[1] != tile_w:
            colored = cv2.resize(colored, (tile_w, tile_h), interpolation=cv2.INTER_LINEAR)
        r = idx // cols
        c = idx % cols
        y0 = r * tile_h
        y1 = y0 + tile_h
        x0 = c * tile_w
        x1 = x0 + tile_w
        grid[y0:y1, x0:x1] = colored
        label = key[:30]
        ts = timestamps.get(key)
        if ts is not None:
            label += f" | t={ts:.2f}"
        cv2.putText(
            grid,
            label,
            (x0 + 10, y0 + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if ordered_keys:
        avg_fps = len(ordered_keys) / max(elapsed, 1e-6)
    else:
        avg_fps = 0.0
    fps_text = f"DepthAnything TRT FPS: {avg_fps:.2f}"
    cv2.putText(
        grid,
        fps_text,
        (20, grid.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    window_title = args.grid_title
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.imshow(window_title, grid)
    if args.grid_save:
        save_path = Path(args.grid_save).expanduser().resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), grid)
        print(f"[info] Saved depth grid to {save_path}")
    print("[info] Press any key to close the depth grid window.")
    cv2.waitKey(0)
    cv2.destroyWindow(window_title)


def run_video_stream(args) -> None:
    video_paths: List[Path] = []
    mode_dir = False
    if args.offline_video_dir:
        mode_dir = True
        video_dir = Path(args.offline_video_dir).expanduser().resolve()
        if not video_dir.exists():
            raise FileNotFoundError(f"Video directory not found: {video_dir}")
        video_paths = _collect_videos(video_dir)
    elif args.offline_video:
        video_path = Path(args.offline_video).expanduser().resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        video_paths = [video_path]
    else:
        raise RuntimeError("run_video_stream requires --offline-video or --offline-video-dir")

    if not video_paths:
        raise RuntimeError("No video streams found for depth inference.")

    if mode_dir and args.max_cams is not None:
        video_paths = video_paths[: max(1, int(args.max_cams))]

    caps: List[cv2.VideoCapture] = []
    names: List[str] = []
    try:
        for path in video_paths:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video {path}")
            caps.append(cap)
            names.append(path.stem)

        engine_path = Path(args.engine).expanduser().resolve()
        with DepthAnythingPool(
            engine_path.parent,
            precision="auto",
            num_workers=max(1, int(args.engine_workers)),
            explicit_engine=engine_path,
        ) as pool:
            frame_count = 0
            window = "DepthAnything TRT"
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            try:
                while True:
                    images: Dict[str, np.ndarray] = {}
                    raw_frames: Dict[str, np.ndarray] = {}
                    for name, cap in zip(names, caps):
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            images = {}
                            break
                        raw_frames[name] = frame
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                        images[name] = rgb
                    if not images:
                        break

                    frame_count += 1
                    if args.grid_limit and args.grid_limit > 0 and frame_count > int(args.grid_limit):
                        break

                    start = time.perf_counter()
                    depths = pool.infer_batch(images)
                    elapsed = time.perf_counter() - start
                    fps = len(images) / max(elapsed, 1e-6)

                    pairs: List[np.ndarray] = []
                    ordered_names = list(images.keys())
                    for name in ordered_names:
                        frame = raw_frames[name]
                        depth = depths.get(name)
                        if depth is None:
                            continue
                        colored = _colorize_depth(depth.astype(np.float32))
                        colored = cv2.resize(colored, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                        pair = np.hstack([frame, colored])
                        cv2.putText(pair, name, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                        pairs.append(pair)

                    if not pairs:
                        continue

                    cols = args.grid_cols if args.grid_cols and args.grid_cols > 0 else len(pairs)
                    cols = max(1, min(cols, len(pairs)))
                    rows = math.ceil(len(pairs) / cols)
                    tile_h, tile_w = pairs[0].shape[:2]
                    canvas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
                    for idx, pair in enumerate(pairs):
                        r = idx // cols
                        c = idx % cols
                        canvas[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = pair
                    cv2.putText(
                        canvas,
                        f"DepthAnything TRT FPS: {fps:.2f}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(window, canvas)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
            finally:
                cv2.destroyWindow(window)
    finally:
        for cap in caps:
            try:
                cap.release()
            except Exception:
                pass


# ---------------------- v4l2 + OpenCV bootstrap ------------------------------
def _run(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        print(f"[v4l2] {' '.join(map(str, cmd))} -> ignored ({(e.stderr or '').strip()})", file=sys.stderr)

def v4l2_preset(dev_index: int, w: int, h: int, fps: int) -> None:
    if shutil.which("v4l2-ctl") is None:
        print("[v4l2] v4l2-ctl not found; skipping hardware preset")
        return
    dev = f"/dev/video{dev_index}"
    _run(["v4l2-ctl", "-d", dev, "--set-fmt-video", f"width={w},height={h},pixelformat=MJPG"])
    _run(["v4l2-ctl", "-d", dev, "--set-parm", str(fps)])
    _run(["v4l2-ctl", "-d", dev, "--set-ctrl", "auto_exposure=3"])
    _run(["v4l2-ctl", "-d", dev, "--set-ctrl", "white_balance_automatic=1"])
    _run(["v4l2-ctl", "-d", dev, "--set-ctrl", "focus_automatic_continuous=1"])
    _run(["v4l2-ctl", "-d", dev, "--set-ctrl", "power_line_frequency=2"])

def open_cam(dev: int, w: int, h: int, fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened(): cap = cv2.VideoCapture(dev)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open /dev/video{dev}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  int(w))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
    cap.set(cv2.CAP_PROP_FPS,          float(fps))
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    try: cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    except Exception: pass
    for _ in range(6): cap.read()
    return cap

# ------------------------ TensorRT 10+ worker thread -------------------------
class TRTDepthWorker(threading.Thread):
    """
    Emits: ((colL, ZL), (colR, ZR), (rawL, rawR))
    where col* are colorized previews (BGR) and Z* are float32 depth (HxW).
    """
    def __init__(self, engine_path: str, side_override: Optional[int]=None, device_index: int = 0):
        super().__init__(daemon=True)
        self.engine_path = engine_path
        self.side_override = side_override
        self.dev_idx = device_index
        self.q_in  = queue.Queue(maxsize=1)
        self.q_out = queue.Queue(maxsize=1)
        self.running = True
        self.err = None
        self.fps = 0.0
        self.side = None

    def submit(self, L: np.ndarray, R: np.ndarray):
        if not self.q_in.full():
            self.q_in.put((L,R))

    def run(self):
        ctx = None
        streamL = streamR = None
        d_inL = d_inR = d_outL = d_outR = None
        h_inL = h_inR = h_outL = h_outR = None
        rt = engine = exeL = exeR = None
        lastColL = None; lastColR = None
        scaleL = RunningScaler(momentum=0.90)
        scaleR = RunningScaler(momentum=0.90)
        try:
            ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
            import pycuda.driver as cuda
            import tensorrt as trt

            cuda.init()
            dev = cuda.Device(self.dev_idx)
            ctx = dev.make_context()

            trt_logger = trt.Logger(trt.Logger.ERROR)
            trt.init_libnvinfer_plugins(trt_logger, "")

            with open(self.engine_path, "rb") as f:
                rt = trt.Runtime(trt_logger)
                engine = rt.deserialize_cuda_engine(f.read())
            if engine is None:
                raise RuntimeError("Failed to deserialize engine.")

            if not hasattr(engine, "num_io_tensors"):
                raise RuntimeError("Requires TensorRT 10+ (I/O tensor API).")

            exeL = engine.create_execution_context()
            exeR = engine.create_execution_context()
            streamL = cuda.Stream(); streamR = cuda.Stream()

            # Discover names
            def discover_io():
                ins, outs = [], []
                for i in range(engine.num_io_tensors):
                    name = engine.get_tensor_name(i)
                    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                        ins.append(name)
                    else:
                        outs.append(name)
                if len(ins) < 1 or len(outs) < 1:
                    raise RuntimeError(f"Unexpected IO tensors: inputs={ins}, outputs={outs}")
                return ins[0], outs[0]
            in_name, out_name = discover_io()

            # Input shape (resolve side)
            in_shape_model = tuple(engine.get_tensor_shape(in_name))
            side = self.side_override or (in_shape_model[2] if (len(in_shape_model) >= 4 and in_shape_model[2] != -1) else 518)
            self.side = int(side)

            runtime_in_shape = (1,3,self.side,self.side)
            exeL.set_input_shape(in_name, runtime_in_shape)
            exeR.set_input_shape(in_name, runtime_in_shape)

            out_shapeL = tuple(exeL.get_tensor_shape(out_name))
            out_shapeR = tuple(exeR.get_tensor_shape(out_name))
            if -1 in out_shapeL or len(out_shapeL) < 4: out_shapeL = (1,1,self.side,self.side)
            if -1 in out_shapeR or len(out_shapeR) < 4: out_shapeR = (1,1,self.side,self.side)

            in_dtype  = trt.nptype(engine.get_tensor_dtype(in_name))
            out_dtype = trt.nptype(engine.get_tensor_dtype(out_name))
            h_inL  = cuda.pagelocked_empty(int(np.prod(runtime_in_shape)), in_dtype)
            h_inR  = cuda.pagelocked_empty(int(np.prod(runtime_in_shape)), in_dtype)
            h_outL = cuda.pagelocked_empty(int(np.prod(out_shapeL)), out_dtype)
            h_outR = cuda.pagelocked_empty(int(np.prod(out_shapeR)), out_dtype)
            d_inL  = cuda.mem_alloc(h_inL.nbytes)
            d_inR  = cuda.mem_alloc(h_inR.nbytes)
            d_outL = cuda.mem_alloc(h_outL.nbytes)
            d_outR = cuda.mem_alloc(h_outR.nbytes)

            evtL_done = cuda.Event(); evtR_done = cuda.Event()

            t0, n = time.time(), 0
            while self.running:
                try: L,R = self.q_in.get(timeout=0.02)
                except queue.Empty: continue

                # LEFT
                try:
                    inpL, padsL, hwL = preprocess(L, self.side)
                    np.copyto(h_inL, inpL.ravel())
                    exeL.set_tensor_address(in_name,  int(d_inL))
                    exeL.set_tensor_address(out_name, int(d_outL))
                    cuda.memcpy_htod_async(d_inL, h_inL, streamL)
                    exeL.execute_async_v3(streamL.handle)
                    cuda.memcpy_dtoh_async(h_outL, d_outL, streamL)
                    evtL_done.record(streamL)
                except Exception:
                    padsL, hwL = (0,0,0,0), L.shape[:2]

                # RIGHT
                try:
                    inpR, padsR, hwR = preprocess(R, self.side)
                    np.copyto(h_inR, inpR.ravel())
                    exeR.set_tensor_address(in_name,  int(d_inR))
                    exeR.set_tensor_address(out_name, int(d_outR))
                    cuda.memcpy_htod_async(d_inR, h_inR, streamR)
                    exeR.execute_async_v3(streamR.handle)
                    cuda.memcpy_dtoh_async(h_outR, d_outR, streamR)
                    evtR_done.record(streamR)
                except Exception:
                    padsR, hwR = (0,0,0,0), R.shape[:2]

                # sync both
                evtL_done.synchronize(); evtR_done.synchronize()

                # Post
                try:
                    depL = h_outL.view(out_dtype).reshape(out_shapeL)
                    vL = depL[np.isfinite(depL)]
                    loL, hiL = scaleL.update(vL)
                    colL = colorize_scaled(depL, padsL, hwL, loL, hiL); 
                    ZL = resize_depth_to_original(depL, padsL, hwL)
                    lastColL = colL
                except Exception:
                    colL = lastColL if lastColL is not None else np.zeros_like(L)
                    ZL = np.zeros(L.shape[:2], np.float32)
                try:
                    depR = h_outR.view(out_dtype).reshape(out_shapeR)
                    vR = depR[np.isfinite(depR)]
                    loR, hiR = scaleR.update(vR)
                    colR = colorize_scaled(depR, padsR, hwR, loR, hiR)
                    ZR = resize_depth_to_original(depR, padsR, hwR)
                    lastColR = colR
                except Exception:
                    colR = lastColR if lastColR is not None else np.zeros_like(R)
                    ZR = np.zeros(R.shape[:2], np.float32)

                if not self.q_out.full():
                    self.q_out.put(((colL, ZL), (colR, ZR), (L, R)))
                n += 1
                now = time.time()
                if now - t0 >= 1.0:
                    self.fps = n/(now-t0); n=0; t0=now

        except Exception as e:
            self.err = f"TRT init/run error: {e}"

        finally:
            try:
                if streamL is not None: streamL.synchronize()
                if streamR is not None: streamR.synchronize()
            except Exception: pass
            for buf in (d_inL, d_inR, d_outL, d_outR):
                try:
                    if buf is not None: buf.free()
                except Exception: pass
            for host in (h_inL, h_inR, h_outL, h_outR):
                try: del host
                except Exception: pass
            try:
                if exeL is not None: del exeL
                if exeR is not None: del exeR
                if engine is not None: del engine
                if rt is not None: del rt
            except Exception: pass
            try:
                if ctx is not None: ctx.pop()
            except Exception: pass

# ------------------------------ ORTHO projector ------------------------------
class OrthoProjector:
    """
    Orthographic point cloud:
      x = pixel x, y = pixel y, z = (depth_norm) * (H/2)
    Uses stride decimation for speed; optional CuPy acceleration.
    """
    def __init__(self, stride: int = 4, use_cupy: bool = True, device: int = 0):
        self.stride = max(1, int(stride))
        self.use_cupy = (_HAS_CUPY and use_cupy)
        self.device = device
        self._xy_cache = {}
        if self.use_cupy:
            cp.cuda.Device(self.device).use()

    def _get_xy(self, H: int, W: int):
        key = (H, W, self.stride, self.use_cupy)
        if key in self._xy_cache:
            return self._xy_cache[key]
        step = self.stride
        xs = np.arange(0, W, step, dtype=np.float32)
        ys = np.arange(0, H, step, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys, indexing='xy')
        if self.use_cupy:
            self._xy_cache[key] = (cp.asarray(xx), cp.asarray(yy))
        else:
            self._xy_cache[key] = (xx, yy)
        return self._xy_cache[key]

    def from_float_depth(self, bgr: np.ndarray, depth_f32: np.ndarray):
        """Return (pts, cols) using per-frame max normalization."""
        H, W = depth_f32.shape[:2]
        step = self.stride
        if self.use_cupy:
            xx, yy = self._get_xy(H, W)
            Zs = cp.asarray(depth_f32[::step, ::step])
            # normalize
            vmax = cp.max(cp.nan_to_num(Zs, nan=0.0))
            vmax = cp.maximum(vmax, 1e-6)
            z = (Zs / vmax) * (H / 2.0)
            valid = cp.isfinite(z) & (z > 0)
            pts = cp.stack((xx[valid], yy[valid], z[valid]), axis=1)
            rgb_small = bgr[::step, ::step, ::-1].astype(np.float32) / 255.0
            col = cp.asarray(rgb_small.reshape(-1,3))[valid.ravel()]
            return cp.asnumpy(pts), cp.asnumpy(col)
        else:
            xx, yy = self._get_xy(H, W)
            Zs = depth_f32[::step, ::step]
            vmax = float(np.nanmax(np.where(np.isfinite(Zs), Zs, 0.0)))
            vmax = max(vmax, 1e-6)
            z = (Zs / vmax) * (H / 2.0)
            valid = np.isfinite(z) & (z > 0)
            pts = np.stack((xx[valid], yy[valid], z[valid]), axis=1)
            rgb_small = bgr[::step, ::step, ::-1].astype(np.float32) / 255.0
            col = rgb_small.reshape(-1,3)[valid.ravel()]
            return pts, col

# ------------------------------ Open3D Viz -----------------------------------
class DualPCDVisualizer:
    def __init__(self, w=900, h=650):
        if not _HAS_O3D:
            raise RuntimeError("Open3D not installed.")
        self.visL = o3d.visualization.Visualizer(); self.visL.create_window("ORTHO Left",  width=w, height=h, left=50,  top=50)
        self.visR = o3d.visualization.Visualizer(); self.visR.create_window("ORTHO Right", width=w, height=h, left=980, top=50)
        self.pcdL = o3d.geometry.PointCloud(); self.pcdR = o3d.geometry.PointCloud()
        self._addedL = False; self._addedR = False

    def update_left(self, pts: np.ndarray, col: np.ndarray):
        self.pcdL.points = o3d.utility.Vector3dVector(pts.astype(np.float64, copy=False))
        self.pcdL.colors = o3d.utility.Vector3dVector(col.astype(np.float64, copy=False))
        if not self._addedL:
            self.visL.add_geometry(self.pcdL)
            self.visL.get_render_option().point_size = 1.5
            self._addedL = True
        else:
            self.visL.update_geometry(self.pcdL)
        self.visL.poll_events(); self.visL.update_renderer()

    def update_right(self, pts: np.ndarray, col: np.ndarray):
        self.pcdR.points = o3d.utility.Vector3dVector(pts.astype(np.float64, copy=False))
        self.pcdR.colors = o3d.utility.Vector3dVector(col.astype(np.float64, copy=False))
        if not self._addedR:
            self.visR.add_geometry(self.pcdR)
            self.visR.get_render_option().point_size = 1.5
            self._addedR = True
        else:
            self.visR.update_geometry(self.pcdR)
        self.visR.poll_events(); self.visR.update_renderer()

    def save(self, outdir: str, ts: str):
        o3d.io.write_point_cloud(os.path.join(outdir, f"left_{ts}.ply"), self.pcdL)
        o3d.io.write_point_cloud(os.path.join(outdir, f"right_{ts}.ply"), self.pcdR)

    def close(self):
        try: self.visL.destroy_window()
        except Exception: pass
        try: self.visR.destroy_window()
        except Exception: pass

# ------------------------------ Main -----------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--offline-dir", type=str, default=None, help="Directory of images to process instead of live capture.")
    ap.add_argument("--offline-video", type=str, default=None, help="Video file to process instead of live capture.")
    ap.add_argument("--offline-video-dir", "--offline_video_dir", type=str, default=None, help="Directory of video streams to process instead of live capture.")
    ap.add_argument("--grid-cols", type=int, default=0, help="Number of columns in offline depth mosaic (0=auto).")
    ap.add_argument("--grid-limit", type=int, default=None, help="Limit number of viewpoints/frames in offline mode (<=0 for unlimited).")
    ap.add_argument("--engine-workers", type=int, default=2, help="Number of TensorRT worker contexts to spawn.")
    ap.add_argument("--max-cams", type=int, default=None, help="Limit number of cameras when streaming a directory of videos.")
    ap.add_argument("--grid-save", type=str, default=None, help="Optional output path for offline depth mosaic.")
    ap.add_argument("--grid-title", type=str, default="Depth Grid", help="Window title for offline visualization.")
    ap.add_argument("--devL", type=int, default=0)
    ap.add_argument("--devR", type=int, default=2)
    ap.add_argument("--w", type=int, default=1920)
    ap.add_argument("--h", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--side", type=int, default=None, help="override model side (e.g., 518)")
    ap.add_argument("--gpu", type=int, default=0, help="CUDA device index")

    # Ortho / performance knobs
    ap.add_argument("--stride", type=int, default=4, help="PCD decimation stride")
    ap.add_argument("--cupy", type=int, default=1, help="1=use CuPy if available")
    ap.add_argument("--pcd_skip", type=int, default=1, help="update PCD every N frames (1=every frame)")
    ap.add_argument("--outliers", type=int, default=0, help="1=run outlier removal (slow)")
    ap.add_argument("--normals", type=int, default=0, help="1=estimate normals (slow)")
    ap.add_argument("--filter_every", type=int, default=15, help="run filters every N frames if enabled")

    # UI
    ap.add_argument("--window_w", type=int, default=900)
    ap.add_argument("--window_h", type=int, default=650)
    ap.add_argument("--disp_scale", type=float, default=0.6)
    ap.add_argument("--save_dir", type=str, default="captures")
    args = ap.parse_args()

    if getattr(args, "offline_video_dir", None) or args.offline_video:
        run_video_stream(args)
        return
    if args.offline_dir:
        run_offline_grid(args)
        return

    os.makedirs(args.save_dir, exist_ok=True)

    # Cameras
    v4l2_preset(args.devL, args.w, args.h, args.fps)
    v4l2_preset(args.devR, args.w, args.h, args.fps)
    capL = open_cam(args.devL, args.w, args.h, args.fps)
    capR = open_cam(args.devR, args.w, args.h, args.fps)

    # TRT worker
    worker = TRTDepthWorker(args.engine, side_override=args.side, device_index=args.gpu)
    worker.start()

    # Ortho projector(s)
    proj = OrthoProjector(stride=args.stride, use_cupy=bool(args.cupy), device=args.gpu)

    # Viz
    viz = DualPCDVisualizer(w=args.window_w, h=args.window_h)
    cv2.namedWindow("RGB | Depth Monitor", cv2.WINDOW_NORMAL)

    frame_idx = 0
    ema=0.0; tprev=time.time()
    last_out = None

    print("[q] quit | [s] snapshot → PNG + PLY | building orthographic clouds…")
    try:
        while True:
            okL, frmL = capL.read(); okR, frmR = capR.read()
            if not okL or not okR or frmL is None or frmR is None:
                time.sleep(0.003)
                continue

            if worker.err:
                print(worker.err); break

            if worker.q_in.empty():
                worker.submit(frmL, frmR)

            while not worker.q_out.empty():
                try:
                    last_out = worker.q_out.get_nowait()
                except queue.Empty:
                    break
            if last_out is None:
                continue

            (colL, ZL), (colR, ZR), (rawL, rawR) = last_out

            # Update PCD at reduced rate if requested
            if (frame_idx % max(1, int(args.pcd_skip))) == 0:
                # ORTHO projection (fast, no filters)
                ptsL, colLrgb = proj.from_float_depth(rawL, ZL)
                ptsR, colRrgb = proj.from_float_depth(rawR, ZR)

                # Optional occasional filtering (slow → run sparsely)
                if args.outliers or args.normals:
                    do_filter = (frame_idx % max(1, args.filter_every)) == 0
                    if do_filter:
                        pcdL = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ptsL))
                        pcdL.colors = o3d.utility.Vector3dVector(colLrgb)
                        pcdR = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ptsR))
                        pcdR.colors = o3d.utility.Vector3dVector(colRrgb)
                        if args.outliers:
                            _, indL = pcdL.remove_statistical_outlier(nb_neighbors=15, std_ratio=1.0)
                            _, indR = pcdR.remove_statistical_outlier(nb_neighbors=15, std_ratio=1.0)
                            pcdL = pcdL.select_by_index(indL); pcdR = pcdR.select_by_index(indR)
                        if args.normals:
                            pcdL.estimate_normals(); pcdL.orient_normals_to_align_with_direction()
                            pcdR.estimate_normals(); pcdR.orient_normals_to_align_with_direction()
                        # extract filtered arrays back
                        ptsL = np.asarray(pcdL.points); colLrgb = np.asarray(pcdL.colors)
                        ptsR = np.asarray(pcdR.points); colRrgb = np.asarray(pcdR.colors)

                viz.update_left(ptsL, colLrgb)
                viz.update_right(ptsR, colRrgb)

            # Monitor grid (scaled)
            try:
                grid = np.vstack([np.hstack([rawL, rawR]), np.hstack([colL, colR])])
                if 0.2 <= args.disp_scale < 1.0:
                    grid = cv2.resize(grid, (int(grid.shape[1]*args.disp_scale), int(grid.shape[0]*args.disp_scale)), interpolation=cv2.INTER_AREA)
                now=time.time(); inst=1/max(now-tprev,1e-6); tprev=now
                ema = inst if ema==0 else 0.2*inst+0.8*ema
                status = f"depthFPS={worker.fps:.1f} | UI {ema:.1f} | stride={args.stride} | cupy={int(proj.use_cupy)}"
                cv2.putText(grid, status, (10,30), 0, 0.9, (255,255,255), 2)
                cv2.imshow("RGB | Depth Monitor", grid)
            except Exception:
                pass

            frame_idx += 1
            k = cv2.waitKey(1) & 0xFF
            if k in (27, ord('q')):
                break
            elif k in (ord('s'), ord('S')):
                ts = time.strftime('%Y%m%d_%H%M%S')
                try:
                    cv2.imwrite(os.path.join(args.save_dir, f"rgbdepth_{ts}.png"), grid)
                    viz.save(args.save_dir, ts)
                    print(f"[save] snapshot → {args.save_dir}")
                except Exception as e:
                    print(f"[save] failed: {e}")

    except KeyboardInterrupt:
        pass
    finally:
        try:
            worker.running = False
            worker.join(timeout=1.0)
        except Exception: pass
        for cap in (capL, capR):
            try: cap.release()
            except Exception: pass
        try:
            viz.close()
            cv2.destroyWindow("RGB | Depth Monitor")
        except Exception: pass
        print("\n[done]")

if __name__ == "__main__":
    main()
