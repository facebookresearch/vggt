#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid MVS (drop-in) — single CUDA context, multi-stream TensorRT, non-blocking loop,
and multithreaded CPU decode (PyAV if available, else OpenCV).

Modes:
  (default) TRT:  t=0 VGGT bootstrap → K,E, metric depth; t>0 DepthAnything (TRT10) per cam, metricized by scale/shift.
  --hf-only:      VGGT per frame (no TRT).
  --bench-trt:    Feed the cached first frame in a tight loop (no decode), to isolate TensorRT throughput.

Key speed choices:
  • ONE primary CUDA context, ONE engine, N exec contexts + N CUDA streams
  • Non-blocking "latest frame" render loop (no 1s stalls)
  • Multithreaded CPU decode (PyAV multi-thread; fallback OpenCV), with bounded prefetch queues
  • Optional pre-letterbox (CPU) in decode threads to offload main thread
  • Square 518×518 path end to end, precomputed unprojection grids
"""

import os, sys, time, argparse, threading, queue, signal
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2

# Optional viz
try:
    import open3d as o3d
    _HAS_O3D = True
except Exception:
    _HAS_O3D = False

# PyAV optional
try:
    import av  # type: ignore
    _HAS_PYAV = True
except Exception:
    _HAS_PYAV = False

# CUDA/TRT optional (needed unless --hf-only)
try:
    import pycuda.driver as cuda  # type: ignore
    import tensorrt as trt        # type: ignore
    _HAS_CUDA_TRT = True
except Exception:
    _HAS_CUDA_TRT = False

# ----------------------------- Utils -----------------------------
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def _to_hw(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr); a = np.squeeze(a)
    if a.ndim == 3 and 1 in a.shape: a = np.squeeze(a)
    if a.ndim != 2:
        if a.ndim == 3 and (a.shape[0]==1 or a.shape[-1]==1): a = np.squeeze(a)
        else: raise ValueError(f"Depth must be 2D after squeeze, got {a.shape}.")
    return a.astype(np.float32, copy=False)

def _preprocess_square_nchw(bgr: np.ndarray, side: int) -> Tuple[np.ndarray, np.ndarray]:
    """Letterbox to square 'side', normalize to NCHW float32; returns (nchw, square_bgr)."""
    H, W = bgr.shape[:2]
    s = min(side / H, side / W)
    nh, nw = int(round(H*s)), int(round(W*s))
    top = (side-nh)//2; left = (side-nw)//2
    bottom = side - nh - top; right = side - nw - left
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    sq_bgr = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
    rgb = cv2.cvtColor(sq_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
    rgb = (rgb - _MEAN)/_STD
    nchw = rgb.transpose(2,0,1)[None].astype(np.float32, copy=False)
    return np.ascontiguousarray(nchw), sq_bgr

def _letterbox_bgr(bgr: np.ndarray, side: int) -> np.ndarray:
    """CPU letterbox to side×side BGR (no normalization)."""
    H, W = bgr.shape[:2]
    s = min(side / H, side / W)
    nh, nw = int(round(H*s)), int(round(W*s))
    top = (side-nh)//2; left = (side-nw)//2
    bottom = side - nh - top; right = side - nw - left
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    sq_bgr = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
    return sq_bgr

def robust_scale_shift(ref: np.ndarray, pred: np.ndarray) -> Tuple[float, float]:
    ref = _to_hw(ref); pred = _to_hw(pred)
    if ref.shape != pred.shape:
        pred = cv2.resize(pred, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_LINEAR)
    m = (ref > 1e-6) & (pred > 1e-6)
    if np.count_nonzero(m) < 100: return 1.0, 0.0
    y = ref[m].reshape(-1,1); x = pred[m].reshape(-1,1)
    A = np.concatenate([x, np.ones_like(x)], axis=1)
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    a0, b0 = float(sol[0,0]), float(sol[1,0])
    r = (A @ np.array([[a0],[b0]]) - y).ravel()
    sigma = 1.4826*np.median(np.abs(r)) + 1e-6
    w = 1.0/np.maximum(1.0, np.abs(r)/(1.345*sigma))
    Aw, yw = A*w[:,None], y*w[:,None]
    sol2, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    return float(sol2[0,0]), float(sol2[1,0])

def precompute_norm_grid(side: int, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y, x = np.meshgrid(np.arange(side, dtype=np.float32),
                       np.arange(side, dtype=np.float32), indexing="ij")
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    return (x - cx)/fx, (y - cy)/fy

# ----------------------------- VGGT helpers -----------------------------
def bootstrap_with_vggt(first_rgb_per_cam: List[np.ndarray]) -> Dict[str, List[np.ndarray]]:
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri as pose_to_mats
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device=="cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    imgs = []
    for im_bgr in first_rgb_per_cam:
        sq_bgr = _letterbox_bgr(im_bgr, 518)
        rgb = cv2.cvtColor(sq_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        imgs.append(rgb.transpose(2,0,1))
    batch = torch.from_numpy(np.stack(imgs)).to(device)  # [S,3,518,518]
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    state = torch.hub.load_state_dict_from_url(_URL, map_location="cpu")
    model.load_state_dict(state); model = model.to(device).eval()
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device=="cuda"), dtype=dtype):
        images5d = batch.unsqueeze(0)           # [1,S,3,518,518]
        tokens, ps_idx = model.aggregator(images5d)
        pose_enc = model.camera_head(tokens)[-1]
        E, K = pose_to_mats(pose_enc, images5d.shape[-2:])
        depth, _ = model.depth_head(tokens, images5d, ps_idx)
    depth_np = depth.squeeze(0).detach().cpu().numpy().astype(np.float32)  # [S,518,518]
    K_np     = K.squeeze(0).detach().cpu().numpy().astype(np.float32)      # [S,3,3]
    E_np     = E.squeeze(0).detach().cpu().numpy().astype(np.float32)      # [S,3,4]
    del model
    if device=="cuda": torch.cuda.empty_cache()
    return {
        "depth_metric": [_to_hw(depth_np[i]) for i in range(depth_np.shape[0])],
        "intrinsic":    [K_np[i] for i in range(K_np.shape[0])],
        "extrinsic":    [E_np[i] for i in range(E_np.shape[0])]
    }

def vggt_per_frame(frames_bgr_sq: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri as pose_to_mats
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device=="cuda" and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
    imgs = []
    for bgr in frames_bgr_sq:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        imgs.append(rgb.transpose(2,0,1))
    batch = torch.from_numpy(np.stack(imgs)).to(device)
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    state = torch.hub.load_state_dict_from_url(_URL, map_location="cpu")
    model.load_state_dict(state); model = model.to(device).eval()
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device=="cuda"), dtype=dtype):
        images5d = batch.unsqueeze(0)           # [1,S,3,518,518]
        tokens, ps_idx = model.aggregator(images5d)
        pose_enc = model.camera_head(tokens)[-1]
        E, K = pose_to_mats(pose_enc, images5d.shape[-2:])
        depth, _ = model.depth_head(tokens, images5d, ps_idx)
    depth_np = depth.squeeze(0).detach().cpu().numpy().astype(np.float32)
    K_np     = K.squeeze(0).detach().cpu().numpy().astype(np.float32)
    E_np     = E.squeeze(0).detach().cpu().numpy().astype(np.float32)
    del model
    if device=="cuda": torch.cuda.empty_cache()
    return ([_to_hw(depth_np[i]) for i in range(depth_np.shape[0])],
            [K_np[i] for i in range(K_np.shape[0])],
            [E_np[i] for i in range(E_np.shape[0])])

# ----------------------------- Decoders -----------------------------
class BaseDecoder(threading.Thread):
    """Decode frames into out_q. If preletterbox=True, output square BGR of size 'side'."""
    def __init__(self, src: str|int, out_q: "queue.Queue[np.ndarray]", stop_evt: threading.Event,
                 preletterbox: bool, side: int, max_fps: Optional[float] = None):
        super().__init__(daemon=True)
        self.src, self.out_q, self.stop_evt = src, out_q, stop_evt
        self.preletterbox, self.side = preletterbox, side
        self.max_fps = max_fps

    def put_frame(self, frame: np.ndarray):
        if self.preletterbox:
            frame = _letterbox_bgr(frame, self.side)
        try:
            if self.out_q.full():
                _ = self.out_q.get_nowait()
            self.out_q.put_nowait(frame)
        except queue.Full:
            pass

class OpenCVDecoder(BaseDecoder):
    def run(self):
        cap = cv2.VideoCapture(self.src)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open source: {self.src}", file=sys.stderr); return
        try: cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception: pass
        last_t = 0.0
        while not self.stop_evt.is_set():
            ok, frame = cap.read()
            if not ok: break
            # optional throttle
            if self.max_fps:
                now = time.time()
                dt = now - last_t
                min_dt = 1.0 / self.max_fps
                if dt < min_dt:
                    time.sleep(min_dt - dt)
                last_t = time.time()
            self.put_frame(frame)
        cap.release()

class PyAVDecoder(BaseDecoder):
    def __init__(self, src: str, out_q: "queue.Queue[np.ndarray]", stop_evt: threading.Event,
                 preletterbox: bool, side: int, threads: int = 4, max_fps: Optional[float] = None):
        super().__init__(src, out_q, stop_evt, preletterbox, side, max_fps)
        self.threads = max(1, int(threads))

    def run(self):
        try:
            container = av.open(self.src)
        except Exception as e:
            print(f"[WARN] PyAV failed to open {self.src}: {e}. Falling back to OpenCV.", file=sys.stderr)
            OpenCVDecoder(self.src, self.out_q, self.stop_evt, self.preletterbox, self.side, self.max_fps).run()
            return

        stream = container.streams.video[0]
        # Try to enable multi-threaded decode
        try:
            # thread_type can be 'AUTO', 'FRAME', or 'SLICE' depending on codec
            stream.thread_type = 'AUTO'
            stream.thread_count = self.threads
        except Exception:
            pass

        last_t = 0.0
        for packet in container.demux(stream):
            if self.stop_evt.is_set(): break
            for frame in packet.decode():
                if self.stop_evt.is_set(): break
                img = frame.to_ndarray(format='bgr24')  # HxWx3 BGR
                if self.max_fps:
                    now = time.time()
                    dt = now - last_t
                    min_dt = 1.0 / self.max_fps
                    if dt < min_dt:
                        time.sleep(min_dt - dt)
                    last_t = time.time()
                self.put_frame(img)
        try:
            container.close()
        except Exception:
            pass

# ----------------------------- TRT scheduler (single context) -----------------------------
class TRTScheduler(threading.Thread):
    """
    Single primary CUDA context; one engine; num_cams execution contexts + streams.
    Pulls frames from per-cam queues; returns (depth_518, sq_bgr_518) per camera.
    If inputs are already letterboxed, we skip the letterbox resize cost.
    """
    def __init__(self, engine_path: str, gpu: int,
                 in_queues: List["queue.Queue[np.ndarray]"],
                 out_queues: List["queue.Queue[Tuple[np.ndarray,np.ndarray]]"],
                 stop_evt: threading.Event,
                 side: int = 518):
        super().__init__(daemon=True)
        self.engine_path, self.gpu = engine_path, gpu
        self.in_qs, self.out_qs = in_queues, out_queues
        self.stop_evt = stop_evt
        self.err: Optional[str] = None
        self.side: int = side
        self.fps: float = 0.0

        # CUDA/TRT resources
        self.pctx = None
        self.rt = None
        self.engine = None
        self.execs = []
        self.streams = []
        self.in_name: Optional[str] = None
        self.out_name: Optional[str] = None
        self.h_in: List[np.ndarray] = []
        self.h_out: List[np.ndarray] = []
        self.d_in: List[Any] = []
        self.d_out: List[Any] = []
        self.out_shapes: List[Tuple[int, ...]] = []

    def _setup(self):
        cuda.init()
        dev = cuda.Device(self.gpu)
        self.pctx = dev.retain_primary_context()
        self.pctx.push()

        logger = trt.Logger(trt.Logger.ERROR)
        trt.init_libnvinfer_plugins(logger, "")
        with open(self.engine_path, "rb") as f:
            self.rt = trt.Runtime(logger)
            self.engine = self.rt.deserialize_cuda_engine(f.read())

        # Query names
        self.in_name  = self.engine.get_tensor_name(0)
        self.out_name = self.engine.get_tensor_name(1)

        # Determine side
        in_shape = self.engine.get_tensor_shape(self.in_name)  # e.g. (-1,3,-1,-1)
        side = max(in_shape[2], in_shape[3])
        if side < 64 or side > 4096 or side < 0: side = self.side
        self.side = int(side)

        num = len(self.in_qs)
        for _ in range(num):
            exe = self.engine.create_execution_context()
            exe.set_input_shape(self.in_name, (1,3,self.side,self.side))
            self.execs.append(exe)
            st = cuda.Stream(); self.streams.append(st)

            out_shape = tuple(exe.get_tensor_shape(self.out_name))
            self.out_shapes.append(out_shape)

            h_in = cuda.pagelocked_empty(1*3*self.side*self.side, np.float32)
            h_out = cuda.pagelocked_empty(int(np.prod(out_shape)), np.float32)
            d_in = cuda.mem_alloc(h_in.nbytes)
            d_out = cuda.mem_alloc(h_out.nbytes)

            exe.set_tensor_address(self.in_name, int(d_in))
            exe.set_tensor_address(self.out_name, int(d_out))

            self.h_in.append(h_in); self.h_out.append(h_out)
            self.d_in.append(d_in); self.d_out.append(d_out)

    def run(self):
        try:
            self._setup()
            last = time.time(); n = 0
            while not self.stop_evt.is_set():
                any_work = False
                for i in range(len(self.in_qs)):
                    if self.out_qs[i].full():  # don't overproduce
                        continue
                    try:
                        frame = self.in_qs[i].get_nowait()
                    except queue.Empty:
                        continue
                    any_work = True

                    # If already square, skip letterbox
                    if frame.shape[0] == self.side and frame.shape[1] == self.side:
                        # Normalize only
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
                        rgb = (rgb - _MEAN)/_STD
                        inp = rgb.transpose(2,0,1)[None].astype(np.float32, copy=False)
                        sq_bgr = frame
                    else:
                        inp, sq_bgr = _preprocess_square_nchw(frame, self.side)

                    np.copyto(self.h_in[i], inp.ravel())

                    st = self.streams[i]; exe = self.execs[i]
                    cuda.memcpy_htod_async(self.d_in[i], self.h_in[i], st)
                    exe.execute_async_v3(st.handle)
                    cuda.memcpy_dtoh_async(self.h_out[i], self.d_out[i], st)
                    st.synchronize()

                    depth_any = self.h_out[i].reshape(self.out_shapes[i])
                    depth_sq = _to_hw(depth_any)
                    try:
                        self.out_qs[i].put_nowait((depth_sq, sq_bgr))
                    except queue.Full:
                        pass
                    n += 1

                if not any_work:
                    time.sleep(0.001)

                now = time.time()
                if now - last >= 1.0:
                    self.fps = n / (now - last); n = 0; last = now
        except Exception as e:
            self.err = f"TRT scheduler error: {e}"
        finally:
            # Free in reverse
            for d in self.d_in:  d.free()
            for d in self.d_out: d.free()
            self.d_in.clear(); self.d_out.clear(); self.h_in.clear(); self.h_out.clear()
            if self.pctx:
                self.pctx.pop()
                try: self.pctx.detach()
                except Exception: pass

# ----------------------------- Orchestration -----------------------------
def run_pipeline(args):
    # OpenCV thread cap
    try: cv2.setNumThreads(max(1, min(os.cpu_count() or 8, args.cv_threads)))
    except Exception: pass

    # Sources
    sources: List[str|int] = []
    if os.path.isdir(args.video_dir):
        vids = sorted([os.path.join(args.video_dir, f) for f in os.listdir(args.video_dir)
                       if f.lower().endswith((".mp4",".mov",".mkv",".avi"))])
        if len(vids) < args.num_cams:
            raise ValueError(f"Found {len(vids)} videos in {args.video_dir}, need {args.num_cams}.")
        sources = vids[:args.num_cams]
    else:
        for tok in args.video_dir.split(","): sources.append(int(tok.strip()))

    # Feeders / decoders
    stop_evt = threading.Event()
    in_qs = [queue.Queue(maxsize=max(1, args.decode_buffer)) for _ in range(args.num_cams)]
    decoders: List[threading.Thread] = []
    side = 518

    def make_decoder(src, out_q):
        if args.decoder == "opencv" or (args.decoder == "auto" and not _HAS_PYAV):
            return OpenCVDecoder(src, out_q, stop_evt, args.preletterbox, side, args.max_fps)
        # PyAV path
        return PyAVDecoder(str(src), out_q, stop_evt, args.preletterbox, side, threads=args.decoder_threads, max_fps=args.max_fps)

    for i in range(args.num_cams):
        dec = make_decoder(sources[i], in_qs[i])
        decoders.append(dec)
        dec.start()

    # Warmup frames (or bench mode: just read one and reuse)
    print("[INFO] Waiting for first frames...")
    first_frames: List[np.ndarray] = []
    if args.bench_trt:
        # Read a single frame from first source using CV (simplify)
        tmp = cv2.VideoCapture(sources[0] if isinstance(sources[0], (str, int)) else str(sources[0]))
        if not tmp.isOpened(): raise RuntimeError(f"Cannot open {sources[0]} for bench.")
        ok, frm = tmp.read(); tmp.release()
        if not ok: raise RuntimeError("Bench-TRT: failed to read a frame.")
        first_frames = [frm for _ in range(args.num_cams)]
    else:
        for i in range(args.num_cams):
            while True:
                try:
                    frm = in_qs[i].get(timeout=5.0)
                    first_frames.append(frm); break
                except queue.Empty:
                    if not decoders[i].is_alive():
                        raise RuntimeError(f"Decoder for {sources[i]} died before providing a frame.")

    print("[INFO] Got first frames for all cameras.")

    # HF-only
    if args.hf_only:
        print("[INFO] HF-only mode: VGGT per frame (no TensorRT).")
        vis = pcd = None
        if args.viz and _HAS_O3D:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="VGGT-only", width=1600, height=900)
            pcd = o3d.geometry.PointCloud()
        t0 = time.perf_counter(); iters = 0; pts_count = 0
        try:
            while True:
                # grab square frames
                sq_frames = []
                for i in range(args.num_cams):
                    try: frm = in_qs[i].get(timeout=1.0)
                    except queue.Empty: frm = first_frames[i]
                    sq = _letterbox_bgr(frm, side)
                    sq_frames.append(sq)
                depths, Ks, Es = vggt_per_frame(sq_frames)

                # build PCD (non-blocking latest not needed here)
                s = max(1, args.pcd_stride)
                all_pts, all_rgb = [], []
                for i in range(args.num_cams):
                    dep = np.clip(depths[i], 0.0, np.percentile(depths[i], 99.9)).astype(np.float32)
                    dep_s = dep[::s, ::s]
                    nx, ny = precompute_norm_grid(dep.shape[0], Ks[i])
                    nx_s, ny_s = nx[::s, ::s], ny[::s, ::s]
                    X = (nx_s * dep_s).ravel(); Y = (ny_s * dep_s).ravel(); Z = dep_s.ravel()
                    pts_cam = np.stack([X, Y, Z], axis=1)
                    Rwc, twc = Es[i][:3,:3], Es[i][:3,3]
                    Rcw = Rwc.T; tcw = - Rcw @ twc
                    pts_w = pts_cam @ Rcw.T + tcw
                    if pts_w.size:
                        zmax = np.percentile(pts_w[:,2], 99.0); m = pts_w[:,2] < zmax; pts_w = pts_w[m]
                    rgb_sq = cv2.cvtColor(sq_frames[i], cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
                    rgb_s  = rgb_sq[::s, ::s, :].reshape(-1,3)
                    if rgb_s.shape[0] >= pts_w.shape[0]: rgb_s = rgb_s[:pts_w.shape[0]]
                    else: rgb_s = np.pad(rgb_s, ((0, pts_w.shape[0]-rgb_s.shape[0]), (0,0)), mode='edge')
                    all_pts.append(pts_w); all_rgb.append(rgb_s)

                if all_pts:
                    pts = np.vstack(all_pts); col = np.vstack(all_rgb); pts_count = pts.shape[0]
                    if args.viz and _HAS_O3D:
                        pcd.points = o3d.utility.Vector3dVector(pts)
                        pcd.colors = o3d.utility.Vector3dVector(col)
                        if iters == 0: vis.add_geometry(pcd)
                        else: vis.update_geometry(pcd)
                        if not vis.poll_events(): break
                        vis.update_renderer()

                iters += 1
                now = time.perf_counter()
                if now - t0 >= 1.0:
                    sys_fps = iters / (now - t0)
                    print(f"[FPS] system={sys_fps:6.2f} | pts={pts_count:,}")
                    t0 = now; iters = 0
        except KeyboardInterrupt:
            print("\n[INFO] Ctrl+C received. Shutting down...")
        finally:
            stop_evt.set(); [d.join(timeout=1.0) for d in decoders]
            if args.viz and _HAS_O3D: vis.destroy_window()
        return

    # ---- TRT mode ----
    if not _HAS_CUDA_TRT:
        raise RuntimeError("PyCUDA/TensorRT not available and --hf-only not set.")

    if args.bench_trt:
        # Letterbox the cached first frames now if preletterbox requested so scheduler can skip extra work
        if args.preletterbox:
            first_frames = [_letterbox_bgr(f, side) for f in first_frames]
        print("[INFO] BENCH-TRT: using cached frames (no decode). Decoders will be stopped.")
        stop_evt.set()
        [d.join(timeout=1.0) for d in decoders]

    print("[INFO] Bootstrapping metric depth + intrinsics/extrinsics...")
    boot = bootstrap_with_vggt(first_frames)
    metric0 = boot["depth_metric"]; Ks = boot["intrinsic"]; Es = boot["extrinsic"]
    print("[INFO] VGGT bootstrap complete.")

    # Scheduler (single context)
    out_qs = [queue.Queue(maxsize=1) for _ in range(args.num_cams)]
    scheduler = TRTScheduler(args.engine, args.gpu, in_qs if not args.bench_trt else [queue.Queue(maxsize=1) for _ in range(args.num_cams)],
                             out_qs, stop_evt, side=side)
    scheduler.start(); time.sleep(0.1)

    # If bench_trt, feed the same prepped frame to each camera queue in a feeder thread
    bench_feeders = []
    if args.bench_trt:
        def _feeder(q, img):
            while not stop_evt.is_set():
                try:
                    if q.full(): _ = q.get_nowait()
                except queue.Empty:
                    pass
                try: q.put_nowait(img)
                except queue.Full: pass
                time.sleep(0.0)
        # re-use first_frames (possibly preletterboxed)
        for i in range(args.num_cams):
            q = scheduler.in_qs[i]
            th = threading.Thread(target=_feeder, args=(q, first_frames[i]), daemon=True)
            th.start(); bench_feeders.append(th)

    # Per-cam scale/shift (re-submit first frame)
    print("[INFO] Calibrating per-camera scale/shift...")
    for i in range(args.num_cams):
        # push a fresh frame for calibration
        if args.bench_trt:
            # already fed by bench_feeders
            pass
        else:
            try:
                if in_qs[i].full(): _ = in_qs[i].get_nowait()
            except queue.Empty: pass
            in_qs[i].put(first_frames[i])

        dep_rel, _ = out_qs[i].get()
        a, b = robust_scale_shift(metric0[i], dep_rel)
        print(f"  cam{i}: scale={a:.4f}, shift={b:.4f}")
        metric0[i] = (a, b)  # store (a,b)

    # Precompute grids
    grids = [precompute_norm_grid(side, Ks[i]) for i in range(args.num_cams)]

    # Viz
    vis = pcd = None
    if args.viz and _HAS_O3D:
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="TRT (single context, multi-stream)", width=1600, height=900)
        pcd = o3d.geometry.PointCloud()

    # Non-blocking, “latest frame” loop
    print("[INFO] Entering real-time loop. Press Ctrl+C to stop.")
    t0 = time.perf_counter()
    iters = 0
    processed_cam_results = 0
    latest: List[Optional[Tuple[np.ndarray, np.ndarray]]] = [None] * args.num_cams
    pts_count = 0

    try:
        while True:
            s = max(1, args.pcd_stride)

            # Drain all available outputs without blocking (no 1s stalls)
            for i in range(args.num_cams):
                if scheduler.err:
                    raise RuntimeError(scheduler.err)
                while True:
                    try:
                        dep_rel, sq_bgr = out_qs[i].get_nowait()
                        latest[i] = (dep_rel, sq_bgr)
                        processed_cam_results += 1
                    except queue.Empty:
                        break

            # Build PCD from whatever is fresh
            all_pts, all_rgb = [], []
            for i in range(args.num_cams):
                if latest[i] is None:
                    continue
                dep_rel, sq_bgr = latest[i]
                a, b = metric0[i]
                dep = a * dep_rel + b
                dep = np.clip(dep, 0.0, np.percentile(dep, 99.9)).astype(np.float32)

                dep_s = dep[::s, ::s]
                nx, ny = grids[i]; nx_s, ny_s = nx[::s, ::s], ny[::s, ::s]
                X = (nx_s * dep_s).ravel(); Y = (ny_s * dep_s).ravel(); Z = dep_s.ravel()
                pts_cam = np.stack([X, Y, Z], axis=1)
                Rwc, twc = Es[i][:3,:3], Es[i][:3,3]
                Rcw = Rwc.T; tcw = - Rcw @ twc
                pts_w = pts_cam @ Rcw.T + tcw
                if pts_w.size:
                    zmax = np.percentile(pts_w[:,2], 99.0); m = pts_w[:,2] < zmax; pts_w = pts_w[m]
                rgb_sq = cv2.cvtColor(sq_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
                rgb_s  = rgb_sq[::s, ::s, :].reshape(-1,3)
                if rgb_s.shape[0] >= pts_w.shape[0]: rgb_s = rgb_s[:pts_w.shape[0]]
                else: rgb_s = np.pad(rgb_s, ((0, pts_w.shape[0]-rgb_s.shape[0]), (0,0)), mode='edge')
                all_pts.append(pts_w); all_rgb.append(rgb_s)

            if all_pts:
                pts = np.vstack(all_pts); col = np.vstack(all_rgb); pts_count = pts.shape[0]
                if args.viz and _HAS_O3D:
                    pcd.points = o3d.utility.Vector3dVector(pts)
                    pcd.colors = o3d.utility.Vector3dVector(col)
                    if iters == 0: vis.add_geometry(pcd)
                    else: vis.update_geometry(pcd)
                    if not vis.poll_events(): return
                    vis.update_renderer()
            else:
                time.sleep(0.001)

            iters += 1
            now = time.perf_counter()
            if now - t0 >= 1.0:
                sys_fps = iters / (now - t0)
                print(f"[FPS] system={sys_fps:6.2f} | trt_sched={scheduler.fps:6.2f} | cam_results/s={processed_cam_results:6.0f} | pts={pts_count:,}")
                t0 = now; iters = 0; processed_cam_results = 0
    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C received. Shutting down...")
    finally:
        stop_evt.set()
        scheduler.join(timeout=2.0)
        [d.join(timeout=1.0) for d in decoders]
        if args.viz and _HAS_O3D: vis.destroy_window()

# ----------------------------- CLI -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Hybrid VGGT bootstrap + DepthAnything v2 (TRT10) — single CUDA context, multi-stream; multithreaded decode; or VGGT-only (--hf-only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("-e","--engine",    type=str, default="", help="DepthAnything v2 TensorRT engine (TRT mode).")
    ap.add_argument("-v","--video-dir", type=str, required=True, help="Dir of N videos, or comma-separated camera indices.")
    ap.add_argument("-n","--num-cams",  type=int, default=4, help="Number of cameras.")
    ap.add_argument("--gpu",            type=int, default=0, help="CUDA device index.")
    ap.add_argument("--viz",            action="store_true", help="Live Open3D visualization.")
    ap.add_argument("--pcd-stride",     type=int, default=2, help="Point-cloud stride decimation.")
    ap.add_argument("--cv-threads",     type=int, default=8, help="OpenCV CPU thread cap.")
    ap.add_argument("--hf-only",        action="store_true", help="Use VGGT per frame (no TRT).")
    ap.add_argument("--bench-trt",      action="store_true", help="Feed cached first frame repeatedly (no decode) to measure pure TRT throughput.")
    # decoder controls
    ap.add_argument("--decoder",        type=str, choices=["auto","pyav","opencv"], default="auto", help="Video decoder backend.")
    ap.add_argument("--decoder-threads",type=int, default=4, help="PyAV: threads per stream.")
    ap.add_argument("--decode-buffer",  type=int, default=8, help="Prefetch buffer size per stream.")
    ap.add_argument("--preletterbox",   action="store_true", help="Letterbox to 518×518 in decoder threads (saves main-thread CPU).")
    ap.add_argument("--max-fps",        type=float, default=None, help="Optional per-stream decode throttle (Hz).")
    args = ap.parse_args()

    if not args.hf_only and not args.engine:
        ap.error("TRT mode requires --engine. Or use --hf-only for VGGT per frame.")
    def _sig(_s,_f): raise KeyboardInterrupt()
    signal.signal(signal.SIGINT, _sig); signal.signal(signal.SIGTERM, _sig)
    run_pipeline(args)

if __name__ == "__main__":
    main()


