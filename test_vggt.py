# test_vggt.py — FastAPI pose server with background VGGT worker (non-blocking HTTP)
# Accepts Unity/HoloLens frames, keeps a sliding window per session, runs VGGT
# on a cadence in a background thread, and exposes /pose/latest + /infer/now.

from __future__ import annotations
import argparse
import io
import json
import os
import shutil
import sys
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from fastapi import FastAPI, File, Form, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# ---------------------------- torch / vggt ----------------------------
import torch
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

# repo-relative import
sys.path.append("vggt/")
from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


# --------------------------- data models ---------------------------
@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion: Optional[List[float]] = None


@dataclass
class Frame:
    frame_id: int
    timestamp_ms: int
    img_rgb: np.ndarray
    K: np.ndarray
    intrinsics: Intrinsics


# ------------------------- utils ----------------------------------
def parse_metadata(meta_json: str) -> Tuple[int, int, Intrinsics, Optional[str]]:
    meta = json.loads(meta_json)
    frame_id = int(meta.get("frame_id", 0))
    timestamp_ms = int(meta.get("timestamp_ms", int(time.time() * 1000)))
    intr = meta.get("intrinsics", {})
    intrinsics = Intrinsics(
        fx=float(intr["fx"]), fy=float(intr["fy"]),
        cx=float(intr["cx"]), cy=float(intr["cy"]),
        width=int(intr["width"]), height=int(intr["height"]),
        distortion=intr.get("distortion") if intr.get("distortion") is not None else None,
    )
    session_id = meta.get("session_id")
    return frame_id, timestamp_ms, intrinsics, session_id


def pil_to_numpy_rgb(img: Image.Image) -> np.ndarray:
    """Force a fresh, contiguous, writeable RGB uint8 array to avoid torch warning."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img, dtype=np.uint8, copy=True)


def make_K(intr: Intrinsics) -> np.ndarray:
    K = np.array([[intr.fx, 0.0, intr.cx],
                  [0.0, intr.fy, intr.cy],
                  [0.0,  0.0,   1.0 ]], dtype=np.float32)
    return K


def save_append_raw(debug_root: str, sid: str, frame_id: int, img_rgb: np.ndarray, meta_json: str) -> None:
    raw_dir = os.path.join(debug_root, sid, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    Image.fromarray(img_rgb).save(os.path.join(raw_dir, f"{frame_id:06d}.jpg"), quality=85, optimize=True)
    with open(os.path.join(raw_dir, f"{frame_id:06d}.json"), "w") as f:
        f.write(meta_json)


def resize_short_side(img: Image.Image, target_short: int) -> Image.Image:
    """Resize with /14 rounding (VGGT patch size is 14)."""
    w, h = img.size

    def r14(x: int) -> int:
        r = int(round(x / 14)) * 14
        return max(14, r)

    if target_short <= 0:
        nw, nh = r14(w), r14(h)
        if (nw, nh) == (w, h):
            return img
        return img.resize((nw, nh), Image.BILINEAR)

    short = min(w, h)
    if short == target_short:
        new_w, new_h = w, h
    else:
        s = target_short / float(short)
        new_w = int(round(w * s))
        new_h = int(round(h * s))

    new_w, new_h = r14(new_w), r14(new_h)
    if (new_w, new_h) == (w, h):
        return img
    return img.resize((new_w, new_h), Image.BILINEAR)


# ----------------------- vggt inference ---------------------------
def _prep_batch_in_memory(
    frames: List[Frame],
    rescale_short_side: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert Frames -> [N,3,H,W] float tensor on device (no disk round-trip)."""
    tensors: List[torch.Tensor] = []
    for fr in frames:
        img = fr.img_rgb
        if rescale_short_side:
            pil = Image.fromarray(img)
            pil = resize_short_side(pil, rescale_short_side)
            img = np.array(pil, dtype=np.uint8, copy=True)

        # ensure contiguous before from_numpy
        img = np.ascontiguousarray(img)
        t = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)
        tensors.append(t)
    batch = torch.stack(tensors, dim=0).to(device, non_blocking=True)
    return batch


def vggt_infer_poses(
    frames: List[Frame],
    device: torch.device,
    model: VGGT,
    autocast_dtype: torch.dtype,
    sid: str,
    debug_save: bool,
    debug_root: str,
    rescale_short_side: int = 0,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    t0 = time.time()
    print(f"[dbg] infer start N={len(frames)} rescale={rescale_short_side}")

    # (A) Optional window snapshot
    if debug_save:
        target_dir = os.path.join(debug_root, sid)
        images_dir = os.path.join(target_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        for f in os.listdir(images_dir):
            fp = os.path.join(images_dir, f)
            try:
                os.remove(fp)
            except Exception:
                pass
        for fr in frames:
            pil = Image.fromarray(fr.img_rgb)
            if rescale_short_side and min(pil.size) != rescale_short_side:
                pil = resize_short_side(pil, rescale_short_side)
            pil.save(os.path.join(images_dir, f"{fr.frame_id:06d}.jpg"), quality=85, optimize=True)

    # (B) In-memory preprocessing
    imgs = _prep_batch_in_memory(frames, rescale_short_side, device)
    print(f"[dbg] prep done imgs={tuple(imgs.shape)} dtype={imgs.dtype} dev={imgs.device}")
    t1 = time.time()

    # (C) Model forward
    use_amp = (device.type == "cuda")
    with torch.inference_mode():
        ctx = torch.cuda.amp.autocast(enabled=use_amp, dtype=autocast_dtype)
        with ctx:
            print("[dbg] forward...")
            predictions = model(imgs)
            print("[dbg] forward done")
    if device.type == "cuda":
        torch.cuda.synchronize()
    print("[dbg] sync done")
    t2 = time.time()

    # (D) Decode pose encodings
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], imgs.shape[-2:])
    extrinsic = extrinsic.detach().cpu().numpy().squeeze(0)

    outs: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(extrinsic.shape[0]):
        Ei = extrinsic[i]
        if Ei.shape == (3, 4):
            H = np.eye(4, dtype=np.float32)
            H[:3, :4] = Ei
        elif Ei.shape == (4, 4):
            H = Ei.astype(np.float32)
        else:
            raise ValueError(f"bad extrinsic shape: {Ei.shape}")
        Twc = np.linalg.inv(H)
        Rwc = Twc[:3, :3].astype(np.float32)
        twc = Twc[:3, 3].astype(np.float32)
        outs.append((Rwc, twc))

    t3 = time.time()
    print(f"[prof] N={len(frames)}  prep={t1-t0:.3f}s  fwd={t2-t1:.3f}s  post={t3-t2:.3f}s  total={t3-t0:.3f}s")
    return outs


# ---------------------- server + sliding window --------------------
class PoseServer:
    def __init__(
        self,
        window_size: int = 4,
        cadence_ms: int = 750,
        debug_save: bool = False,
        debug_root: str = "sessions",
        inference_short_side: int = 252,
    ):
        self.window_size = window_size
        self.cadence_ms = cadence_ms
        self.buffers: Dict[str, Deque[Frame]] = {}
        self.last_run_ms: Dict[str, int] = {}
        self.inference_busy: Dict[str, bool] = {}
        self.debug_save = debug_save
        self.debug_root = debug_root
        self.inference_short_side = inference_short_side

        self.default_session = "default"

        # load vggt once
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[info] torch.cuda.is_available={torch.cuda.is_available()}")
        print(f"[info] torch.version.cuda={getattr(torch.version, 'cuda', None)}  cudnn={torch.backends.cudnn.version()}")
        if self.device.type != "cuda":
            raise RuntimeError("CUDA not available in this process. Please run on a CUDA-capable machine.")

        self.model = VGGT()
        url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        sd = torch.hub.load_state_dict_from_url(url, map_location="cpu")
        self.model.load_state_dict(sd)
        self.model.eval().to(self.device)

        cap = torch.cuda.get_device_capability()
        is_a100 = (cap[0] == 8 and cap[1] == 0)
        is_hopper = (cap[0] >= 9)
        self.autocast_dtype = torch.bfloat16 if (is_a100 or is_hopper) else torch.float16
        print(f"[info] device={torch.cuda.get_device_name(0)}  cap={cap}  autocast_dtype={self.autocast_dtype}")

        # Warmup (build kernels, autotune)
        H = max(128, self.inference_short_side or 252)
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=self.autocast_dtype):
            dummy = torch.randn(2, 3, H, H, device=self.device)
            _ = self.model(dummy)
            _ = self.model(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        print("[info] warmup done")

        # Background worker
        self.latest_pose: Dict[str, Dict] = {}
        self._stop = False
        self._worker = threading.Thread(target=self._background_loop, daemon=True)
        self._worker.start()
        print("[info] background inference worker started")

    # session helpers
    def _get_session(self, session_id: Optional[str]) -> str:
        sid = session_id or self.default_session
        if sid not in self.buffers:
            self.buffers[sid] = deque(maxlen=self.window_size)
            self.last_run_ms[sid] = 0
            self.inference_busy[sid] = False
        return sid

    def reset_session(self, session_id: Optional[str]) -> None:
        sid = session_id or self.default_session
        self.buffers[sid] = deque(maxlen=self.window_size)
        self.last_run_ms[sid] = 0
        self.inference_busy[sid] = False
        print(f"[info] reset session '{sid}'.")

    def add_frame(self, sid: str, frame: Frame) -> None:
        self.buffers[sid].append(frame)

    # throttle
    def _should_run(self, sid: str, now_ms: int) -> bool:
        if self.inference_busy.get(sid, False):
            return False
        return (now_ms - self.last_run_ms.get(sid, 0)) >= self.cadence_ms

    # background worker loop
    def _background_loop(self):
        """Run VGGT per session at cadence without blocking HTTP."""
        while not self._stop:
            now_ms = int(time.time() * 1000)
            for sid in list(self.buffers.keys()):
                try:
                    if self.inference_busy.get(sid, False):
                        continue
                    if len(self.buffers[sid]) < 2:
                        continue
                    if not self._should_run(sid, now_ms):
                        continue

                    self.inference_busy[sid] = True
                    frames = list(self.buffers[sid])
                    t0 = time.time()

                    poses = vggt_infer_poses(
                        frames=frames,
                        device=self.device,
                        model=self.model,
                        autocast_dtype=self.autocast_dtype,
                        sid=sid,
                        debug_save=self.debug_save,
                        debug_root=self.debug_root,
                        rescale_short_side=self.inference_short_side,
                    )
                    R, t = poses[-1]
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                    t1 = time.time()

                    latest = frames[-1]
                    dt_infer = (t1 - t0) * 1000.0
                    self.last_run_ms[sid] = int(time.time() * 1000)
                    self.inference_busy[sid] = False

                    # Build Twc (camera->world) and K
                    Twc = np.eye(4, dtype=np.float32)
                    Twc[:3, :3] = R
                    Twc[:3,  3] = t
                    K = np.array([
                        [latest.intrinsics.fx, 0.0,                 latest.intrinsics.cx],
                        [0.0,                  latest.intrinsics.fy, latest.intrinsics.cy],
                        [0.0,                  0.0,                 1.0]
                    ], dtype=np.float32)

                    # store latest result for API consumers
                    self.latest_pose[sid] = {
                        "frame_id": latest.frame_id,
                        "timestamp_ms": latest.timestamp_ms,
                        "buffer": len(frames),
                        "R": R.tolist(),
                        "t": t.tolist(),
                        "Twc": Twc.tolist(),
                        "K": K.tolist(),
                        "inference_ms": dt_infer,
                    }

                    # nice terminal print
                    print("\n" + "-" * 64)
                    print(f"[{time.strftime('%H:%M:%S')}] session={sid} frame={latest.frame_id} buffer={len(frames)}")
                    print(f"vggt inference: {dt_infer:.1f} ms")
                    np.set_printoptions(precision=3, suppress=True)
                    x, y, z = np.array(t, dtype=float).ravel()
                    print(f"position_world_xyz = ({x:.3f}, {y:.3f}, {z:.3f})  m")
                    print("R=\n" + str(R))
                    print("t= " + str(t) + "  (m, up-to-scale unless metric anchored)")
                    print("-" * 64)
                except Exception as e:
                    print(f"[worker][{sid}] error: {e}")
                    self.inference_busy[sid] = False

            time.sleep(max(0.001, self.cadence_ms / 1000.0))

    # optional: manual one-shot inference (synchronous) for debugging
    def run_once(self, sid: str) -> Optional[Dict]:
        if len(self.buffers.get(sid, [])) < 2:
            return None
        if self.inference_busy.get(sid, False):
            return None
        self.inference_busy[sid] = True
        try:
            frames = list(self.buffers[sid])
            t0 = time.time()
            poses = vggt_infer_poses(
                frames=frames, device=self.device, model=self.model,
                autocast_dtype=self.autocast_dtype, sid=sid,
                debug_save=self.debug_save, debug_root=self.debug_root,
                rescale_short_side=self.inference_short_side,
            )
            R, t = poses[-1]
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            dt = (time.time() - t0) * 1000.0
            latest = frames[-1]

            Twc = np.eye(4, dtype=np.float32)
            Twc[:3, :3] = R
            Twc[:3,  3] = t
            K = np.array([
                [latest.intrinsics.fx, 0.0,                 latest.intrinsics.cx],
                [0.0,                  latest.intrinsics.fy, latest.intrinsics.cy],
                [0.0,                  0.0,                 1.0]
            ], dtype=np.float32)

            out = {
                "frame_id": latest.frame_id,
                "timestamp_ms": latest.timestamp_ms,
                "buffer": len(frames),
                "R": R.tolist(),
                "t": t.tolist(),
                "Twc": Twc.tolist(),
                "K": K.tolist(),
                "inference_ms": dt,
            }
            self.latest_pose[sid] = out
            return out
        finally:
            self.last_run_ms[sid] = int(time.time() * 1000)
            self.inference_busy[sid] = False


# ------------------------------ api -------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

SERVER: PoseServer
_server: Optional[PoseServer] = None


@app.on_event("startup")
async def _on_startup():
    global _server
    if _server is None:
        _server = PoseServer(window_size=4, cadence_ms=750, debug_save=False, inference_short_side=252)
        print("[ready] pose server initialized (defaults: window=4, cadence=750ms, short_side=252, debug_save=False)")


@app.post("/frame")
async def post_frame(
    image: UploadFile = File(...),
    metadata: str = Form(...),
):
    assert _server is not None

    # parse metadata
    try:
        frame_id, timestamp_ms, intrinsics, session_id = parse_metadata(metadata)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"bad metadata: {e}"})
    sid = _server._get_session(session_id)

    # decode image
    try:
        raw = await image.read()
        pil = Image.open(io.BytesIO(raw))
        img_rgb = pil_to_numpy_rgb(pil)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"bad image: {e}"})

    # optional: save every upload
    if _server.debug_save:
        try:
            save_append_raw(_server.debug_root, sid, frame_id, img_rgb, metadata)
        except Exception as e:
            print(f"[warn] save raw failed: {e}")

    # build camera matrix (not used by VGGT forward, but kept)
    K = make_K(intrinsics)

    # push into sliding window
    frame = Frame(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        img_rgb=img_rgb,
        K=K,
        intrinsics=intrinsics,
    )
    _server.add_frame(sid, frame)

    # non-blocking: just acknowledge receipt
    print(f"[recv] sid={sid} frame={frame_id} buffer={len(_server.buffers[sid])}")
    return JSONResponse(status_code=202, content={"status": "queued"})


@app.get("/pose/latest")
async def get_latest_pose(session_id: Optional[str] = Query(None)):
    assert _server is not None
    sid = _server._get_session(session_id)
    pose = _server.latest_pose.get(sid)
    if pose is None:
        return {"status": "none"}
    return {"status": "ok", "pose": pose}


@app.post("/infer/now")
async def infer_now(session_id: Optional[str] = Query(None)):
    assert _server is not None
    sid = _server._get_session(session_id)
    out = _server.run_once(sid)
    if out is None:
        return {"status": "none"}
    return {"status": "ok", "pose": out}


@app.post("/reset")
async def reset_session(payload: Dict[str, str]):
    assert _server is not None
    _server.reset_session(payload.get("session_id"))
    return {"status": "reset"}


@app.post("/tag")
async def tag_event(payload: Dict):
    """
    optional: record a 'point tap' sent from unity
    payload: {"session_id":"hl2","point":[x,y,z],"label":"tap","timestamp_ms":...}
    """
    assert _server is not None
    sid = payload.get("session_id") or _server.default_session
    pt = payload.get("point")
    ts = payload.get("timestamp_ms", int(time.time() * 1000))
    label = payload.get("label", "tap")

    if not isinstance(pt, list) or len(pt) != 3:
        return JSONResponse(status_code=400, content={"error": "point must be [x,y,z]"})

    tag_dir = os.path.join(_server.debug_root, sid)
    os.makedirs(tag_dir, exist_ok=True)
    with open(os.path.join(tag_dir, "tags.jsonl"), "a") as f:
        f.write(json.dumps({"timestamp_ms": ts, "label": label, "point": pt}) + "\n")

    print(f"[tag] {sid}  {label}  point={pt}")
    return {"status": "ok"}


# ------------------------------ main ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--cadence-ms", type=int, default=750)
    parser.add_argument("--short-side", type=int, default=252,
                        help="rescale short side before vggt; 0 keeps original; multiples of 14 recommended (e.g., 252/266/280)")
    parser.add_argument("--debug-save", action="store_true", help="keep per-session files under sessions/<sid>/")
    parser.add_argument("--debug-root", type=str, default="sessions")
    args = parser.parse_args()

    global _server
    _server = PoseServer(
        window_size=args.window,
        cadence_ms=args.cadence_ms,
        debug_save=args.debug_save,
        debug_root=args.debug_root,
        inference_short_side=args.short_side,
    )
    print(f"[ready] pose server initialized (window={args.window}, cadence={args.cadence_ms}ms, short_side={args.short_side}, debug_save={args.debug_save})")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", workers=1)


if __name__ == "__main__":
    main()
