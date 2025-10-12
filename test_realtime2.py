# test_realtime2.py
# FastAPI server: camera-only VGGT (AR-preserving), sliding window, single-flight inference.

from __future__ import annotations
import argparse
import asyncio
import io
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from PIL import Image

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Use the camera-only engine you just built
from test_cameras_only import CameraOnlyVGGT

# --------------------------- config ---------------------------

@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    window: int = 4          # frames per inference
    size: int = 320          # long-side cap (AR preserved)
    debug_root: str = "sessions"  # where we store per-session images
    allow_origins: list[str] = None

CFG = ServerConfig()
GLOBAL_LOCK = asyncio.Lock()  # single-flight across all sessions

# --------------------------- sessions -------------------------

@dataclass
class SessionState:
    sid: str
    engine: CameraOnlyVGGT
    images_dir: str
    next_frame_id: int = 0

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
    _sessions[sid] = st
    return st

def reset_session(sid: Optional[str]) -> None:
    sid = sid or "default"
    if sid in _sessions:
        # Drop the old engine/queue by recreating the session state
        del _sessions[sid]
    # Also wipe the on-disk folder (best-effort)
    try:
        base = os.path.join(CFG.debug_root, sid)
        if os.path.isdir(base):
            for root, _, files in os.walk(base, topdown=False):
                for f in files:
                    try:
                        os.remove(os.path.join(root, f))
                    except Exception:
                        pass
    except Exception:
        pass

# --------------------------- api ------------------------------

app = FastAPI(title="VGGT Camera-Only Realtime")
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

@app.post("/reset")
async def reset(payload: Dict[str, str]):
    reset_session(payload.get("session_id"))
    return {"status": "reset"}

@app.post("/frame")
async def post_frame(
    image: UploadFile = File(...),
    metadata: str = Form("{}"),
):
    """
    Accept exactly one frame.
    Behavior:
      - If an inference is in progress anywhere, reject with 409 (no file is saved).
      - Otherwise: save image under sessions/<sid>/images/, push to session engine,
        run a single camera-only inference on the current window, and return the
        CURRENT frame's pose + timings.
    """
    # Hard "busy" gate: do not accept while computing
    if GLOBAL_LOCK.locked():
        return JSONResponse(status_code=409, content={"status": "busy", "detail": "inference in progress"})

    # Parse minimal metadata (session_id and optional frame_id; others are ignored here)
    try:
        meta = json.loads(metadata) if metadata else {}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"bad metadata json: {e}"})
    sid = meta.get("session_id") or "default"
    frame_id_from_client = meta.get("frame_id")

    # Create/get session
    sess = get_or_create_session(sid)

    # Acquire the single-flight lock and do all work inside
    async with GLOBAL_LOCK:
        # Read and save the image to disk (engine consumes file paths)
        try:
            raw = await image.read()
            pil = Image.open(io.BytesIO(raw))
            if pil.mode != "RGB":
                pil = pil.convert("RGB")
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"bad image: {e}"})

        # Choose filename
        if isinstance(frame_id_from_client, int):
            fid = int(frame_id_from_client)
            # keep server-side counter ahead of the largest seen
            sess.next_frame_id = max(sess.next_frame_id, fid + 1)
        else:
            fid = sess.next_frame_id
            sess.next_frame_id += 1

        img_path = os.path.join(sess.images_dir, f"{fid:06d}.jpg")
        try:
            pil.save(img_path, quality=92)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"failed to save image: {e}"})

        # Push path into the sliding window and run a single-pass inference
        t0 = time.time()
        sess.engine.push_path(img_path)
        res = sess.engine.infer_latest()
        dt_ms = (time.time() - t0) * 1000.0

        latest = res["latest"]           # camera for the newest frame
        timings = res["timings"]
        timings["request_total_ms"] = round(dt_ms, 2)
        buffer_len = timings.get("num_frames", 0)

        # Log to server console for quick eyeballing
        pos = latest["position_m"]; rpy = latest["euler_xyz_deg"]
        print(f"[{time.strftime('%H:%M:%S')}] sid={sid} frame={fid} "
              f"buf={buffer_len}  pos=({pos['x']:.3f},{pos['y']:.3f},{pos['z']:.3f}) m  "
              f"rpy=({rpy['roll']:.1f},{rpy['pitch']:.1f},{rpy['yaw']:.1f})  "
              f"total={timings['total_s']}s  infer={timings['inference_s']}s")

        return {
            "status": "ok",
            "session_id": sid,
            "frame_id": fid,
            "buffer": buffer_len,
            "latest": latest,
            "timings": timings,
        }

# --------------------------- main -------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=CFG.host)
    parser.add_argument("--port", type=int, default=CFG.port)
    parser.add_argument("--window", type=int, default=CFG.window, help="frames per inference")
    parser.add_argument("--size", type=int, default=CFG.size, help="long-side cap (AR preserved)")
    parser.add_argument("--debug-root", type=str, default=CFG.debug_root)
    args = parser.parse_args()

    CFG.host = args.host
    CFG.port = args.port
    CFG.window = args.window
    CFG.size = args.size
    CFG.debug_root = args.debug_root

    print(f"[ready] camera-only server  window={CFG.window}  size_cap={CFG.size}  root={CFG.debug_root}")
    uvicorn.run(app, host=CFG.host, port=CFG.port, log_level="warning")

if __name__ == "__main__":
    main()
