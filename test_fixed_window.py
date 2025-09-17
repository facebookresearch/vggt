 # test_fixed_window.py
# Fixed-window (1 anchor + last N) camera-only VGGT pose server
# - First frame becomes the anchor.
# - Subsequent frames fill a deque of size (window-1).
# - Once >=2 frames exist, we run inference and choose a fixed world
#   normalization (origin) exactly once. We report has_origin to the client.
#
# Endpoints
#   GET  /health                → basic info
#   POST /reset {session_id}    → wipe a session (anchor, recents, origin)
#   POST /frame (multipart)     → { image: UploadFile, metadata: JSON string }
#                                 returns latest camera pose + has_origin
#   GET  /status?session_id=... → session status (anchor set? buffer size? has_origin?)
#
# Notes
# - Requires test_cameras_only.CameraOnlyVGGT with .infer_paths(paths, world_norm_T=None, fix_gauge_once=True)
#   returning a dict containing:
#       cameras_world : List[dict]  # per image, already in the fixed world if world_norm_T was provided
#       world_norm_T  : Optional[List[List[float]]] 4x4 chosen normalization (on first multi-view solve)
#       timings       : Optional[dict]               e.g. {"total_s": float, "inference_s": float}
#
# Thomas-ready:
# - Adds "has_origin" and "announce_origin" (true only on the first time origin becomes available).
# - Sphere/arrow logic on Unity can simply watch has_origin to toggle visibility.

from __future__ import annotations
import argparse
import asyncio
import io
import json
import os
import time
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, List
from collections import deque

import numpy as np
from PIL import Image

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Your camera-only engine
from test_cameras_only import CameraOnlyVGGT

# --------------------------- config ---------------------------

@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    window: int = 4          # total frames per inference (1 anchor + (window-1) recents). min=2
    size: int = 320          # long-side cap (AR preserved)
    debug_root: str = "sessions"
    allow_origins: Optional[List[str]] = None

CFG = ServerConfig()
GLOBAL_LOCK = asyncio.Lock()  # single-flight guard across all sessions

# --------------------------- sessions -------------------------

@dataclass
class SessionState:
    sid: str
    engine: CameraOnlyVGGT
    images_dir: str
    next_frame_id: int = 0

    # Selection + fixed-world normalization
    anchor_path: Optional[str] = None
    anchor_frame_id: Optional[int] = None
    recent: Deque[str] = field(default_factory=lambda: deque(maxlen=3))  # set maxlen in factory then override
    world_norm_T: Optional[np.ndarray] = None  # 4x4 transform chosen once and reused
    origin_frame_id: Optional[int] = None      # frame at which world_norm_T was established

_sessions: Dict[str, SessionState] = {}

def get_or_create_session(sid: Optional[str]) -> SessionState:
    sid = sid or "default"
    if sid in _sessions:
        return _sessions[sid]
    os.makedirs(CFG.debug_root, exist_ok=True)
    images_dir = os.path.join(CFG.debug_root, sid, "images")
    os.makedirs(images_dir, exist_ok=True)
    engine = CameraOnlyVGGT(size=CFG.size, window=CFG.window)
    st = SessionState(sid=sid, engine=engine, images_dir=images_dir)
    # enforce "fixed window" recent capacity = window-1, min 1
    st.recent = deque(maxlen=max(1, CFG.window - 1))
    _sessions[sid] = st
    return st


def _wipe_dir_tree(path: str) -> None:
    # best-effort wipe (files only) to keep dirs
    try:
        if os.path.isdir(path):
            for root, _, files in os.walk(path, topdown=False):
                for f in files:
                    try:
                        os.remove(os.path.join(root, f))
                    except Exception:
                        pass
    except Exception:
        pass


def reset_session(sid: Optional[str]) -> None:
    sid = sid or "default"
    if sid in _sessions:
        del _sessions[sid]
    _wipe_dir_tree(os.path.join(CFG.debug_root, sid))

# --------------------------- app ------------------------------

app = FastAPI(title="VGGT Camera-Only (Fixed Anchor + Last-N)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CFG.allow_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok", "window": CFG.window, "size_cap": CFG.size}

@app.get("/status")
async def status(session_id: Optional[str] = None):
    st = _sessions.get(session_id or "default")
    if st is None:
        return {"status": "absent", "session_id": session_id or "default"}
    return {
        "status": "ok",
        "session_id": st.sid,
        "has_anchor": st.anchor_path is not None,
        "buffer": (1 if st.anchor_path else 0) + len(st.recent),
        "has_origin": st.world_norm_T is not None,
        "origin_frame_id": st.origin_frame_id,
        "anchor_frame_id": st.anchor_frame_id,
    }

@app.post("/reset")
async def reset(payload: Dict[str, str]):
    reset_session(payload.get("session_id"))
    return {"status": "reset"}

# --------------------------- /frame ---------------------------

@app.post("/frame")
async def post_frame(
    image: UploadFile = File(...),
    metadata: str = Form("{}"),
):
    """
    Accept exactly one frame.
    Behavior:
      - Reject with 409 if any inference is in-flight (global single-flight).
      - Save the image, update [anchor + last (window-1)].
      - If <2 total frames: return identity pose + has_origin=False.
      - Otherwise: run engine on [anchor, recents], keep a fixed world by reusing world_norm_T,
        set it once on first multi-view solve, and return the latest camera pose.
    """
    if GLOBAL_LOCK.locked():
        return JSONResponse(status_code=409, content={"status": "busy", "detail": "inference in progress"})

    # Parse metadata
    try:
        meta = json.loads(metadata) if metadata else {}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"bad metadata json: {e}"})

    sid = meta.get("session_id") or "default"
    client_frame_id = meta.get("frame_id")
    sess = get_or_create_session(sid)

    async with GLOBAL_LOCK:
        # --- decode & save ---
        try:
            raw = await image.read()
            pil = Image.open(io.BytesIO(raw))
            if pil.mode != "RGB":
                pil = pil.convert("RGB")
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"bad image: {e}"})

        if isinstance(client_frame_id, int):
            fid = int(client_frame_id)
            sess.next_frame_id = max(sess.next_frame_id, fid + 1)
        else:
            fid = sess.next_frame_id
            sess.next_frame_id += 1

        img_path = os.path.join(sess.images_dir, f"{fid:06d}.jpg")
        try:
            pil.save(img_path, quality=92)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"failed to save image: {e}"})

        # --- anchor + recents ---
        if sess.anchor_path is None:
            sess.anchor_path = img_path
            sess.anchor_frame_id = fid
        else:
            sess.recent.append(img_path)

        batch_paths = ([sess.anchor_path] if sess.anchor_path else []) + list(sess.recent)
        latest_index = len(batch_paths) - 1

        # --- if not enough views yet, identity + has_origin=False ---
        if len(batch_paths) < 2:
            latest = {
                "position_m": {"x": 0.0, "y": 0.0, "z": 0.0},
                "euler_xyz_deg": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
                "matrix_c2w": np.eye(4, dtype=np.float32).tolist(),
                "frame_path": img_path,
            }
            timings = {"total_s": 0.0, "inference_s": 0.0, "num_frames": len(batch_paths)}
            has_origin = False
            print(f"[{time.strftime('%H:%M:%S')}] sid={sid} frame={fid} buf={len(batch_paths)} "
                  f"pos=(0.000,0.000,0.000) m  rpy=(0.0,0.0,0.0)  origin={has_origin}  total=0.0s")
            return {
                "status": "ok",
                "session_id": sid,
                "frame_id": fid,
                "buffer": len(batch_paths),
                "has_origin": has_origin,
                "announce_origin": False,
                "latest": latest,
                "timings": timings,
            }

        # --- run inference on [anchor, last up to (window-1)] ---
        if not hasattr(sess.engine, "infer_paths"):
            return JSONResponse(
                status_code=500,
                content={
                    "error": (
                        "CameraOnlyVGGT.infer_paths(...) is required. "
                        "Please paste the infer_paths implementation into test_cameras_only.py."
                    )
                },
            )

        t0 = time.time()
        res = sess.engine.infer_paths(
            batch_paths,
            world_norm_T=sess.world_norm_T,  # None until first multi-view solve
            fix_gauge_once=True,             # choose world_norm_T once, then reuse
        )
        req_ms = (time.time() - t0) * 1000.0

        # Persist world normalization once
        announce_origin = False
        if sess.world_norm_T is None and res.get("world_norm_T") is not None:
            sess.world_norm_T = np.array(res["world_norm_T"], dtype=np.float32)
            sess.origin_frame_id = fid
            announce_origin = True

        cams = res["cameras_world"]
        latest_cam = cams[latest_index]

        timings = res.get("timings", {})
        timings["request_total_ms"] = round(req_ms, 2)
        timings["num_frames"] = len(batch_paths)

        pos = latest_cam.get("position_m", {})
        rpy = latest_cam.get("euler_xyz_deg", {})
        px, py, pz = pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0)
        rr, rp, ry = rpy.get("roll", 0.0), rpy.get("pitch", 0.0), rpy.get("yaw", 0.0)
        has_origin = sess.world_norm_T is not None
        print(f"[{time.strftime('%H:%M:%S')}] sid={sid} frame={fid} buf={len(batch_paths)}  "
              f"pos=({px:.3f},{py:.3f},{pz:.3f}) m  rpy=({rr:.1f},{rp:.1f},{ry:.1f})  "
              f"origin={has_origin}{'(!)' if announce_origin else ''}  "
              f"total={timings.get('total_s', 0.0)}s  infer={timings.get('inference_s', 0.0)}s")

        out = {
            "status": "ok",
            "session_id": sid,
            "frame_id": fid,
            "buffer": len(batch_paths),
            "has_origin": has_origin,
            "announce_origin": announce_origin,
            "latest": latest_cam,
            "timings": timings,
        }
        # Only include the transform the moment we first pick it (handy for debugging).
        if announce_origin:
            out["world_norm_T"] = sess.world_norm_T.tolist()
            out["origin_frame_id"] = sess.origin_frame_id
            out["anchor_frame_id"] = sess.anchor_frame_id
        return out

# --------------------------- main -------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=CFG.host)
    parser.add_argument("--port", type=int, default=CFG.port)
    parser.add_argument(
        "--window",
        type=int,
        default=CFG.window,
        help="total frames per inference (anchor + recents); min=2; recents = window-1",
    )
    parser.add_argument("--size", type=int, default=CFG.size, help="long-side cap (AR preserved)")
    parser.add_argument("--debug-root", type=str, default=CFG.debug_root)
    args = parser.parse_args()

    CFG.host = args.host
    CFG.port = args.port
    CFG.window = max(2, args.window)
    CFG.size = args.size
    CFG.debug_root = args.debug_root

    print(
        f"[ready] fixed-window server  window={CFG.window} (anchor+{CFG.window-1})  "
        f"size_cap={CFG.size}  root={CFG.debug_root}"
    )
    uvicorn.run(app, host=CFG.host, port=CFG.port, log_level="info")

if __name__ == "__main__":
    main() # Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
