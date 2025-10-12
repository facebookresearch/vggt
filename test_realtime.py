# test_realtime.py
# fastapi server that accepts hololens frames, runs vggt on a sliding window,
# prints latest pose, optionally rescales to a target short side,
# saves all uploads under sessions/<sid>/raw/, and supports simple /tag events.

from __future__ import annotations
import argparse
import io
import json
import os
import shutil
import sys
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# vggt imports (adjust if your repo layout differs)
import torch
torch.set_float32_matmul_precision("high")          # mild speedup on win
torch.backends.cudnn.benchmark = True               # helps on stable input sizes

sys.path.append("vggt/")
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
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
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.asarray(img)


def make_K(intr: Intrinsics) -> np.ndarray:
    K = np.array([[intr.fx, 0.0, intr.cx],
                  [0.0, intr.fy, intr.cy],
                  [0.0,  0.0,   1.0 ]], dtype=np.float32)
    return K


def save_append_raw(debug_root: str, sid: str, frame_id: int, img_rgb: np.ndarray, meta_json: str) -> None:
    raw_dir = os.path.join(debug_root, sid, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    Image.fromarray(img_rgb).save(os.path.join(raw_dir, f"{frame_id:06d}.jpg"))
    with open(os.path.join(raw_dir, f"{frame_id:06d}.json"), "w") as f:
        f.write(meta_json)


def resize_short_side(img: Image.Image, target_short: int) -> Image.Image:
    w, h = img.size
    if target_short <= 0:
        return img
    short = min(w, h)
    if short == target_short:
        return img
    scale = target_short / float(short)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    # round to multiples of 8 to be friendly with jpg/conv
    new_w = (new_w + 7) & ~7
    new_h = (new_h + 7) & ~7
    return img.resize((new_w, new_h), Image.BILINEAR)


# ----------------------- vggt inference ---------------------------
def vggt_infer_poses(
    frames: List[Frame],
    device: torch.device,
    model: VGGT,
    autocast_dtype: torch.dtype,
    sid: str,
    debug_save: bool,
    debug_root: str,
    rescale_short_side: int = 0,   # 0 = keep as-is; try 256/320 for speed
) -> List[Tuple[np.ndarray, np.ndarray]]:
    # 1) materialize images to disk (temp or debug folder)
    if debug_save:
        target_dir = os.path.join(debug_root, sid)
        images_dir = os.path.join(target_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        # show only current window
        for f in os.listdir(images_dir):
            try:
                os.remove(os.path.join(images_dir, f))
            except Exception:
                pass
    else:
        target_dir = tempfile.mkdtemp(prefix=f"vggt_rt_{sid}_")
        images_dir = os.path.join(target_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

    image_paths: List[str] = []
    for fr in frames:
        pil = Image.fromarray(fr.img_rgb)
        if rescale_short_side and min(pil.size) != rescale_short_side:
            pil = resize_short_side(pil, rescale_short_side)
        fn = os.path.join(images_dir, f"{fr.frame_id:06d}.png")
        pil.save(fn)
        image_paths.append(fn)

    try:
        # 2) preprocess and run model
        t0 = time.time()
        imgs = load_and_preprocess_images(image_paths).to(device)
        with torch.no_grad():
            use_amp = (device.type == "cuda")
            ctx = torch.cuda.amp.autocast(enabled=use_amp, dtype=autocast_dtype)
            with ctx:
                predictions = model(imgs)

        # 3) decode pose encoding to extrinsics/intrinsics
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], imgs.shape[-2:])
        extrinsic = extrinsic.detach().cpu().numpy().squeeze(0)

        # 4) convert per-frame to camera->world (R,t)
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
        return outs
    finally:
        if not debug_save:
            shutil.rmtree(target_dir, ignore_errors=True)


# ---------------------- server + sliding window --------------------
class PoseServer:
    def __init__(
        self,
        window_size: int = 4,
        cadence_ms: int = 750,
        debug_save: bool = False,
        debug_root: str = "sessions",
        inference_short_side: int = 0,   
    ):
        self.window_size = window_size
        self.cadence_ms = cadence_ms
        self.buffers: Dict[str, Deque[Frame]] = {}
        self.last_run_ms: Dict[str, int] = {}
        self.inference_busy: Dict[str, bool] = {}
        self.debug_save = debug_save
        self.debug_root = debug_rootf
        self.inference_short_side = inference_short_side

        self.default_session = "default"

        # load vggt once
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[info] using device: {self.device}")
        self.model = VGGT()
        url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        sd = torch.hub.load_state_dict_from_url(url, map_location=self.device)
        self.model.load_state_dict(sd)
        self.model.eval().to(self.device)
        cc = torch.cuda.get_device_capability()[0] if torch.cuda.is_available() else 0
        self.autocast_dtype = torch.bfloat16 if cc >= 8 else torch.float16

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
        if self.inference_busy[sid]:
            return False
        return (now_ms - self.last_run_ms[sid]) >= self.cadence_ms

    # trigger inference
    def maybe_run_inference(self, sid: str) -> Optional[Dict]:
        now_ms = int(time.time() * 1000)
        if not self._should_run(sid, now_ms):
            return None
        if len(self.buffers[sid]) < 2:
            return None

        self.inference_busy[sid] = True
        t0 = time.time()
        frames = list(self.buffers[sid])

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
        assert len(poses) == len(frames)
        R, t = poses[-1]

        t1 = time.time()
        self.last_run_ms[sid] = int(now_ms)
        self.inference_busy[sid] = False

        latest = frames[-1]
        dt_infer = (t1 - t0) * 1000.0

        # terminal print
        print("\n" + "-" * 64)
        print(f"[{time.strftime('%H:%M:%S')}] session={sid} frame={latest.frame_id} buffer={len(frames)}")
        print(f"vggt inference: {dt_infer:.1f} ms")
        np.set_printoptions(precision=3, suppress=True)
        x, y, z = t.astype(float).ravel()
        print(f"position_world_xyz = ({x:.3f}, {y:.3f}, {z:.3f})  m")
        print("R=\n" + str(R))
        print("t= " + str(t) + "  (m, up-to-scale unless metric anchored)")
        print("-" * 64)

        return {
            "frame_id": latest.frame_id,
            "timestamp_ms": latest.timestamp_ms,
            "buffer": len(frames),
            "R": R.tolist(),
            "t": t.tolist(),
            "inference_ms": dt_infer,
        }


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
        _server = PoseServer(window_size=4, cadence_ms=750, debug_save=True, inference_short_side=256)
        print("[ready] pose server initialized (defaults: window=4, cadence=750ms, short_side=256, debug_save=True)")


@app.post("/frame")
async def post_frame(
    image: UploadFile = File(...),
    metadata: str = Form(...),
):
    assert _server is not None

    #parse metadata
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

    # append-only archive of every upload
    if _server.debug_save:
        try:
            save_append_raw(_server.debug_root, sid, frame_id, img_rgb, metadata)
        except Exception as e:
            print(f"[warn] save raw failed: {e}")

    # build camera matrix (not currently used by vggt forward, but kept for reference)
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

    # maybe run vggt
    out = _server.maybe_run_inference(sid)
    if out is not None:
        return JSONResponse(status_code=200, content={"status": "ok", "pose": out})
    else:
        return JSONResponse(status_code=202, content={"status": "queued"})


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
    parser.add_argument("--short-side", type=int, default=256, help="rescale short side before vggt; 0 keeps original")
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

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
