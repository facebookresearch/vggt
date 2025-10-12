# test_fixed_anchor_shared.py
# Shared-anchor, fixed-window camera-only VGGT pose server
# - The FIRST frame from device_id == "EditorCam" becomes the immutable anchor for the session.
# - All subsequent inference for ANY device in that session always includes [ANCHOR + last (window-1) frames of that device].
# - has_origin flips true once we have [anchor + >=1 more frame] so poses are meaningful.
#
# Endpoints
#   GET  /health
#   POST /reset            { session_id }            → clears anchor + buffers
#   POST /frame            multipart { image, metadata-json } → returns pose + has_origin
#   GET  /status?session_id=...                      → status snapshot
#
# Notes
# - Plug your existing VGGT "camera-only" runner where indicated (run_vggt()).
# - We store the anchor as sessions/<sid>/anchor.jpg (so it's stable across device joins).
# - window refers to total frames per inference: 1 anchor + (window-1) recents of the posting device.

from __future__ import annotations
import argparse
import io
import json
import os
import threading
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime

import numpy as np
from PIL import Image

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# --------------------- Config & CLI ---------------------

def parse_args():
    p = argparse.ArgumentParser("shared-anchor camera-only server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--window", type=int, default=8, help="total frames per inference (1 anchor + window-1 recents)")
    p.add_argument("--size", type=int, default=540, help="target short-side resize (server-side)")
    p.add_argument("--root", default="sessions", help="root folder for saved frames")
    p.add_argument("--anchor_device", default="EditorCam", help="device_id that is allowed to set the session anchor")
    return p.parse_args()

ARGS = parse_args()
os.makedirs(ARGS.root, exist_ok=True)

# --------------------- Simple image utils ---------------------

def load_image_from_upload(upload: UploadFile) -> Image.Image:
    data = upload.file.read()
    return Image.open(io.BytesIO(data)).convert("RGB")

def resize_short_side(im: Image.Image, target_short: int) -> Image.Image:
    w, h = im.size
    short = min(w, h)
    if short <= target_short:
        return im
    scale = target_short / short
    nw, nh = int(round(w * scale)), int(round(h * scale))
    return im.resize((nw, nh), Image.BILINEAR)

def pil_to_numpy(im: Image.Image) -> np.ndarray:
    return np.array(im, dtype=np.uint8)

def save_jpg(im: Image.Image, path: str, quality: int = 90):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    im.save(path, format="JPEG", quality=quality)

# --------------------- VGGT stub (plug yours) ---------------------
# Replace this with your actual runner: returns (Rcw: 3x3 np, tcw: 3,)
def run_vggt(frames_rgb: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    frames_rgb: list of HxWx3 uint8 (first is the anchor, remaining are recents of posting device)
    Return: (Rcw (3x3 float64), tcw (3,))
    """
    # TODO: integrate your CameraOnlyVGGT here.
    # For now, return identity pose (camera at world origin) to keep plumbing intact.
    R = np.eye(3, dtype=np.float64)
    t = np.zeros(3, dtype=np.float64)
    return R, t

# --------------------- Session state ---------------------

@dataclass
class DeviceBuffer:
    frames: Deque[np.ndarray] = field(default_factory=deque)  # recent frames for this device

@dataclass
class SessionState:
    lock: threading.Lock = field(default_factory=threading.Lock)
    anchor_img: Optional[np.ndarray] = None
    anchor_path: Optional[str] = None
    device_buffers: Dict[str, DeviceBuffer] = field(default_factory=dict)
    has_origin: bool = False  # flips true once anchor+>=1 frame exist

# session_id -> state
SESSIONS: Dict[str, SessionState] = {}

def get_session(sid: str) -> SessionState:
    if sid not in SESSIONS:
        SESSIONS[sid] = SessionState()
    return SESSIONS[sid]

# --------------------- FastAPI app ---------------------

app = FastAPI(title="VGGT Shared-Anchor Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "window": ARGS.window, "size_cap": ARGS.size, "root": ARGS.root, "anchor_device": ARGS.anchor_device}

@app.post("/reset")
def reset(payload: Dict):
    sid = payload.get("session_id", "")
    if not sid:
        return JSONResponse({"error": "missing session_id"}, status_code=400)
    SESSIONS.pop(sid, None)
    # wipe folder
    sess_dir = os.path.join(ARGS.root, sid)
    if os.path.isdir(sess_dir):
        try:
            for fname in os.listdir(sess_dir):
                fpath = os.path.join(sess_dir, fname)
                try: os.remove(fpath)
                except: pass
        except: pass
    return {"ok": True, "session_id": sid}

@app.get("/status")
def status(session_id: str):
    st = SESSIONS.get(session_id)
    if not st:
        return {"exists": False}
    with st.lock:
        counts = {d: len(buf.frames) for d, buf in st.device_buffers.items()}
        return {
            "exists": True,
            "has_anchor": st.anchor_img is not None,
            "has_origin": st.has_origin,
            "device_buffers": counts,
            "anchor_path": st.anchor_path,
        }

@app.post("/frame")
def frame(image: UploadFile = File(...), metadata: str = Form(...)):
    try:
        meta = json.loads(metadata)
    except Exception:
        return JSONResponse({"error": "bad metadata json"}, status_code=400)

    sid: str = meta.get("session_id", "")
    did: str = meta.get("device_id", "")
    if not sid or not did:
        return JSONResponse({"error": "missing session_id or device_id"}, status_code=400)

    st = get_session(sid)

    # load + resize image
    try:
        im = load_image_from_upload(image)
    except Exception as e:
        return JSONResponse({"error": f"decode image failed: {e}"}, status_code=400)
    im = resize_short_side(im, ARGS.size)
    np_im = pil_to_numpy(im)  # HxWx3

    sess_dir = os.path.join(ARGS.root, sid)
    os.makedirs(sess_dir, exist_ok=True)

    with st.lock:
        # 1) Ensure per-device buffer
        if did not in st.device_buffers:
            st.device_buffers[did] = DeviceBuffer()

        # 2) If no anchor yet, only the configured anchor_device can set it
        if st.anchor_img is None:
            if did == ARGS.anchor_device:
                st.anchor_img = np_im.copy()
                # persist for visibility/debug
                anchor_path = os.path.join(sess_dir, "anchor.jpg")
                save_jpg(im, anchor_path, quality=92)
                st.anchor_path = anchor_path
                # Note: has_origin not true yet until we also have a second frame to pair with
            else:
                # Non-anchor device posted before anchor exists: we still accept its frame (buffer it),
                # but we won't run inference yet and has_origin remains False.
                buf = st.device_buffers[did].frames
                buf.append(np_im)
                _cap_deque(buf, ARGS.window - 1)  # keep capacity
                return {
                    "session_id": sid,
                    "device_id": did,
                    "has_origin": False,
                    "Rcw": None, "tcw": None,
                    "note": f"waiting for anchor from {ARGS.anchor_device}"
                }

        # 3) Anchor exists: push this frame to device’s buffer
        buf = st.device_buffers[did].frames
        buf.append(np_im)
        _cap_deque(buf, ARGS.window - 1)  # keep at most (window-1) recents per device

        # 4) Determine if we have enough to infer: anchor + >=1 recent
        if st.anchor_img is None or len(buf) == 0:
            return {
                "session_id": sid,
                "device_id": did,
                "has_origin": False,
                "Rcw": None, "tcw": None
            }

        # 5) Build inference window = [ANCHOR] + list(buf)
        frames_rgb = [st.anchor_img] + list(buf)

        # 6) Run VGGT (replace stub with your real call)
        Rcw, tcw = run_vggt(frames_rgb)

        # 7) Flip has_origin true the first time we produce a pose from anchor+≥1
        st.has_origin = True

        # 8) Save raw for debugging (optional)
        ts = meta.get("timestamp") or datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        save_jpg(im, os.path.join(sess_dir, f"{did}_{ts}.jpg"), quality=88)

        # 9) Reply
        return {
            "session_id": sid,
            "device_id": did,
            "has_origin": True,
            "Rcw": _mat_to_rowmajor_list(Rcw),
            "tcw": tcw.tolist(),
        }

def _cap_deque(dq: Deque, cap: int):
    while len(dq) > cap:
        dq.popleft()

def _mat_to_rowmajor_list(M: np.ndarray) -> List[float]:
    # 3x3 → 9 floats row-major
    return [float(M[0,0]), float(M[0,1]), float(M[0,2]),
            float(M[1,0]), float(M[1,1]), float(M[1,2]),
            float(M[2,0]), float(M[2,1]), float(M[2,2])]

# --------------------- main ---------------------

if __name__ == "__main__":
    print(f"[ready] shared-anchor server  window={ARGS.window}  size_cap={ARGS.size}  root={ARGS.root}  anchor_device={ARGS.anchor_device}")
    uvicorn.run(app, host=ARGS.host, port=ARGS.port)
