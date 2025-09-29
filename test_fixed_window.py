from __future__ import annotations
import argparse
import asyncio
import io
import json
import os
import time
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, List, Tuple
from collections import deque

import numpy as np
from PIL import Image

from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from test_cameras_only import CameraOnlyVGGT

# --------------------------- config ---------------------------

@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    window: int = 4                               # total per device: (anchor + recents)
    size: int = 320                               # long-side cap (AR preserved by engine)
    debug_root: str = "sessions"
    allow_origins: Optional[List[str]] = None

CFG = ServerConfig()
GLOBAL_LOCK = asyncio.Lock()

# --------------------------- multi-device structures ----------

@dataclass
class DeviceTrack:
    device_id: str
    images_dir: str
    next_frame_id: int = 0
    anchor_path: Optional[str] = None
    anchor_frame_id: Optional[int] = None
    recent: Deque[str] = field(default_factory=lambda: deque(maxlen=3))
    intrinsics: Optional[List[float]] = None      # [fx,fy,cx,cy] or None

    def alloc_frame_id(self, client_frame_id: Optional[int]) -> int:
        if isinstance(client_frame_id, int):
            fid = int(client_frame_id)
            self.next_frame_id = max(self.next_frame_id, fid + 1)
            return fid
        fid = self.next_frame_id
        self.next_frame_id += 1
        return fid

@dataclass
class SessionState:
    sid: str
    engine: CameraOnlyVGGT
    root_dir: str
    anchor_device: Optional[str] = None           # first device to submit becomes anchor
    world_norm_T: Optional[np.ndarray] = None     # ONLY for the anchor device/session world
    origin_frame_id: Optional[Tuple[str,int]] = None  # (device_id, frame_id) when origin fixed
    devices: Dict[str, DeviceTrack] = field(default_factory=dict)
    xforms: Dict[str, np.ndarray] = field(default_factory=dict)  # key: f"{dev}->anchor"
    has_origin: bool = False  # [PATCH] session-level flag once any valid pose produced

_sessions: Dict[str, SessionState] = {}
_ip_to_device: Dict[str, str] = {}
_device_to_ip: Dict[str, str] = {}

def _session_dir(sid: str) -> str:
    return os.path.join(CFG.debug_root, sid)

def _device_dir(sid: str, device_id: str) -> str:
    return os.path.join(_session_dir(sid), f"dev_{device_id}")

def get_or_create_session(sid: Optional[str]) -> SessionState:
    sid = sid or "default"
    if sid in _sessions:
        return _sessions[sid]
    os.makedirs(CFG.debug_root, exist_ok=True)
    root_dir = _session_dir(sid)
    os.makedirs(root_dir, exist_ok=True)
    engine = CameraOnlyVGGT(size=CFG.size, window=CFG.window)
    st = SessionState(sid=sid, engine=engine, root_dir=root_dir)
    _sessions[sid] = st
    return st

def ensure_device(sess: SessionState, device_id: str, intrinsics: Optional[List[float]]) -> DeviceTrack:
    if device_id not in sess.devices:
        ddir = _device_dir(sess.sid, device_id)
        os.makedirs(ddir, exist_ok=True)
        track = DeviceTrack(device_id=device_id, images_dir=ddir)
        # set device deque maxlen = window-1 for recents
        track.recent = deque(maxlen=max(1, CFG.window - 1))
        if intrinsics is not None:
            track.intrinsics = intrinsics
        sess.devices[device_id] = track
        if sess.anchor_device is None:
            sess.anchor_device = device_id  # first device becomes anchor
    else:
        track = sess.devices[device_id]
        if track.intrinsics is None and intrinsics is not None:
            track.intrinsics = intrinsics
    return track

def _wipe_dir_tree(path: str) -> None:
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
    _wipe_dir_tree(_session_dir(sid))

def as_4x4(M) -> np.ndarray:
    M = np.array(M, dtype=np.float64)
    if M.shape == (16,):
        M = M.reshape(4,4)
    assert M.shape == (4,4), f"expected 4x4, got {M.shape}"
    return M

def matmul(A, B) -> np.ndarray:
    return as_4x4(A) @ as_4x4(B)

def _flat16(M: np.ndarray) -> List[float]:
    M = np.array(M, dtype=np.float32)
    return [
        float(M[0,0]), float(M[0,1]), float(M[0,2]), float(M[0,3]),
        float(M[1,0]), float(M[1,1]), float(M[1,2]), float(M[1,3]),
        float(M[2,0]), float(M[2,1]), float(M[2,2]), float(M[2,3]),
        float(M[3,0]), float(M[3,1]), float(M[3,2]), float(M[3,3]),
    ]


# --------------------------- app ------------------------------

app = FastAPI(title="VGGT Camera-Only (Multi-Device: Fixed Anchor + Per-Device Last-N)")
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
    per_dev = {}
    for did, tr in st.devices.items():
        per_dev[did] = {
            "has_anchor": tr.anchor_path is not None,
            "buffer": (1 if tr.anchor_path else 0) + len(tr.recent),
            "next_frame_id": tr.next_frame_id
        }
    return {
        "status": "ok",
        "session_id": st.sid,
        "anchor_device": st.anchor_device,
        "devices": list(st.devices.keys()),
        "per_device": per_dev,
        "has_origin_anchor_world": st.world_norm_T is not None,
        "origin_fixed_at": st.origin_frame_id,  # (device_id, frame_id)
        "alignments": list(st.xforms.keys()),   # e.g., ["HL2->anchor"]
        "has_origin": st.has_origin,            # [PATCH] expose session-level flag
    }

@app.post("/reset")
async def reset(payload: Dict[str, str]):
    reset_session(payload.get("session_id"))
    return {"status": "reset"}

# --------------------------- alignment ------------------------

@app.post("/align/set")
async def align_set(payload: Dict):
    """
    Set mapping from a device's world to the anchor's world.
    payload = { "session_id": "...", "device_id": "HL2", "T_anchor_worldD": [[...4x4...]] }
    """
    sid = payload.get("session_id", "default")
    dev = payload.get("device_id")
    T = payload.get("T_anchor_worldD")
    if dev is None or T is None:
        return JSONResponse(status_code=400, content={"error":"need device_id and T_anchor_worldD"})
    if sid not in _sessions:
        return JSONResponse(status_code=400, content={"error":"unknown session"})
    sess = _sessions[sid]
    if sess.anchor_device is None:
        return JSONResponse(status_code=400, content={"error":"no anchor_device set yet"})
    key = f"{dev}->anchor"
    sess.xforms[key] = as_4x4(T).astype(np.float32)
    return {"ok": True, "session_id": sid, "device_id": dev, "key": key}

@app.get("/align/get")
async def align_get(session_id: str, device_id: str):
    sess = _sessions.get(session_id)
    if not sess:
        return JSONResponse(status_code=400, content={"error":"unknown session"})
    key = f"{device_id}->anchor"
    T = sess.xforms.get(key)
    if T is None:
        return {"has_alignment": False, "anchor_device": sess.anchor_device}
    return {"has_alignment": True, "anchor_device": sess.anchor_device, "T_anchor_worldD": T.tolist()}

# --------------------------- /frame ---------------------------

@app.post("/frame")
async def post_frame(
    request: Request,
    image: UploadFile = File(...),
    metadata: str = Form("{}"),
):
    """
    Expected metadata JSON (min):
        {
          "session_id": "room-001",
          "device_id": "HL1",
          "frame_id": 12,                 # optional (server will allocate if absent)
          "intrinsics": [fx,fy,cx,cy]     # optional (recommended)
        }
    """
    if GLOBAL_LOCK.locked():
        return JSONResponse(status_code=409, content={"status": "busy", "detail": "inference in progress"})

    try:
        meta = json.loads(metadata) if metadata else {}
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"bad metadata json: {e}"})

    sid = meta.get("session_id") or "default"
    device_id = meta.get("device_id") or "default"
    client_frame_id = meta.get("frame_id")
    intrinsics = meta.get("intrinsics")  # [fx,fy,cx,cy] or None

    client_ip = request.client.host if request and request.client else "unknown"
    norm = (device_id or "").strip()
    if client_ip in _ip_to_device:
        if _ip_to_device[client_ip] != norm:
            norm = _ip_to_device[client_ip]
    else:
        _ip_to_device[client_ip] = norm
    if norm in _device_to_ip:
        if _device_to_ip[norm] != client_ip:
            pass
    else:
        _device_to_ip[norm] = client_ip
    device_id = norm

    sess = get_or_create_session(sid)
    track = ensure_device(sess, device_id, intrinsics)

    async with GLOBAL_LOCK:
        # decode & save image under dev-specific folder
        try:
            raw = await image.read()
            pil = Image.open(io.BytesIO(raw))
            if pil.mode != "RGB":
                pil = pil.convert("RGB")
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"bad image: {e}"})

        fid = track.alloc_frame_id(client_frame_id)

        if device_id == "EditorCam" : 
            img_path = os.path.join(track.images_dir, f"+{fid:06d}.jpg")
        else :
            img_path = os.path.join(track.images_dir, f"{fid:06d}.jpg")

        try:
            pil.save(img_path, quality=92)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"failed to save image: {e}"})

        # build this device's window 
        if track.anchor_path is None:
            track.anchor_path = img_path
            track.anchor_frame_id = fid
        else:
            track.recent.append(img_path)

        batch_paths = ([track.anchor_path] if track.anchor_path else []) + list(track.recent)
        latest_index = len(batch_paths) - 1

        # If not enough frames for this device yet, return identity + session has_origin flag
        if len(batch_paths) < 2:
            latest = {
                "position_m": {"x": 0.0, "y": 0.0, "z": 0.0},
                "euler_xyz_deg": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
                "matrix_c2w": np.eye(4, dtype=np.float32).tolist(),
                "frame_path": img_path,
            }
            timings = {"total_s": 0.0, "inference_s": 0.0, "num_frames": len(batch_paths)}
            has_origin = sess.has_origin  #session-level, not per-device/anchor-only
            print(f"[{time.strftime('%H:%M:%S')}] sid={sid} dev={device_id} frame={fid} buf={len(batch_paths)} "
                  f"pos=(0.000,0.000,0.000) m  rpy=(0.0,0.0,0.0)  origin={has_origin}  total=0.0s")
            #add Rcw/tcw in all replies for client compatibility (identity here)
            Rcw = np.eye(3, dtype=np.float32)
            tcw = np.zeros(3, dtype=np.float32)
            return {
                "status": "ok",
                "session_id": sid,
                "device_id": device_id,
                "frame_id": fid,
                "buffer": len(batch_paths),
                "has_origin": has_origin,
                "need_calibration": False,
                "announce_origin": False,
                "latest": latest,
                "pos": [0.0, 0.0, 0.0],
                "rpy": [0.0, 0.0, 0.0],
                "Rcw": _mat3_to_rowmajor_list(Rcw),   
                "tcw": tcw.tolist(),                  
                "timings": timings,
                "anchor_device": sess.anchor_device,
                "latest_c2w_rowmajor16": _flat16(np.eye(4, dtype=np.float32)),  # identity
            }

        #run VGGT on THIS device's window
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

        #For the anchor device, pass the session's world_norm_T to keep gauge fixed once.
        world_norm_for_infer = sess.world_norm_T if device_id == sess.anchor_device else None

        t0 = time.time()
        res = sess.engine.infer_paths(
            batch_paths,
            world_norm_T=world_norm_for_infer,
            fix_gauge_once=True,
        )
        req_ms = (time.time() - t0) * 1000.0

        #If this is the anchor device and world not fixed yet, capture it now
        announce_origin = False
        if device_id == sess.anchor_device and sess.world_norm_T is None and res.get("world_norm_T") is not None:
            sess.world_norm_T = np.array(res["world_norm_T"], dtype=np.float32)
            sess.origin_frame_id = (device_id, fid)
            announce_origin = True

        cams = res["cameras_world"]
        latest_cam = cams[latest_index]

        #Extract c2w (4x4) if provided
        c2w = latest_cam.get("matrix_c2w", None)
        if c2w is None:
            c2w = np.eye(4, dtype=np.float32).tolist()

        timings = res.get("timings", {})
        timings["request_total_ms"] = round(req_ms, 2)
        timings["num_frames"] = len(batch_paths)

        pos = latest_cam.get("position_m", {})
        rpy = latest_cam.get("euler_xyz_deg", {})
        px, py, pz = pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0)
        rr, rp, ry = rpy.get("roll", 0.0), rpy.get("pitch", 0.0), rpy.get("yaw", 0.0)

        #mapping to ANCHOR world for the response
        mapped_c2w = np.array(c2w, dtype=np.float32)
        need_calib = False

        #Decide has_origin at the session level after we've produced any valid pose
        #We set it TRUE below after inference (this is only happening once per session).
        #but for mapping: if non-anchor & alignment exists, map into anchor world; otherwise leave as-is.
        if device_id != sess.anchor_device:
            key = f"{device_id}->anchor"
            if key in sess.xforms:
                T_anchor_worldD = sess.xforms[key]  # 4x4
                mapped_c2w = matmul(T_anchor_worldD, mapped_c2w).astype(np.float32)
            else:
                need_calib = True  #pose is valid but still in device-local world

        #Once we got here, we *did* compute a pose , so mark session has_origin True.
        sess.has_origin = True

        #derive Rcw/tcw from mapped_c2w (W_T_C): top-left 3x3, top-right 3x1
        Rcw = mapped_c2w[:3, :3].astype(np.float32)
        tcw = mapped_c2w[:3, 3].astype(np.float32)

        mapped_px, mapped_py, mapped_pz = float(mapped_c2w[0,3]), float(mapped_c2w[1,3]), float(mapped_c2w[2,3])
        has_origin = sess.has_origin  

        print(f"[{time.strftime('%H:%M:%S')}] sid={sid} dev={device_id} frame={fid} buf={len(batch_paths)}  "
              f"pos=({px:.3f},{py:.3f},{pz:.3f}) m  rpy=({rr:.1f},{rp:.1f},{ry:.1f})  "
              f"origin={has_origin}{'(!)' if announce_origin else ''}  "
              f"total={timings.get('total_s', 0.0)}s  infer={timings.get('inference_s', 0.0)}s")

        out = {
            "status": "ok",
            "session_id": sid,
            "device_id": device_id,
            "anchor_device": sess.anchor_device,
            "frame_id": fid,
            "buffer": len(batch_paths),
            "has_origin": has_origin,          #this is session level
            "need_calibration": need_calib,    #still tells you if alignment missing for mapping
            "announce_origin": announce_origin,
            "latest": {
                **latest_cam,
                "matrix_c2w": mapped_c2w.tolist(),  #possibly mapped into ANCHOR world
            },
            "pos": [mapped_px, mapped_py, mapped_pz],
            "rpy": [float(rr), float(rp), float(ry)],
            "Rcw": _mat3_to_rowmajor_list(Rcw),   #for Unity client compatibility
            "tcw": tcw.tolist(),                  
            "timings": timings,
        }

        out["latest_c2w_rowmajor16"] = _flat16(mapped_c2w)

        if announce_origin:
            out["world_norm_T"] = sess.world_norm_T.tolist()
            out["origin_frame_id"] = {"device_id": device_id, "frame_id": fid}
            out["anchor_frame_id"] = {"device_id": device_id, "frame_id": track.anchor_frame_id}
        return out

# --------------------------- helpers -------------------------------

def _mat3_to_rowmajor_list(M: np.ndarray) -> List[float]:
    M = np.asarray(M)
    return [float(M[0,0]), float(M[0,1]), float(M[0,2]),
            float(M[1,0]), float(M[1,1]), float(M[1,2]),
            float(M[2,0]), float(M[2,1]), float(M[2,2])]

# --------------------------- main -------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=CFG.host)
    parser.add_argument("--port", type=int, default=CFG.port)
    parser.add_argument(
        "--window",
        type=int,
        default=CFG.window,
        help="per-device frames per inference (anchor + recents); min=2; recents = window-1",
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
        f"[ready] multi-device fixed-window server  per-device window={CFG.window} (anchor+{CFG.window-1})  "
        f"size_cap={CFG.size}  root={CFG.debug_root}"
    )
    uvicorn.run(app, host=CFG.host, port=CFG.port, log_level="info")

if __name__ == "__main__":
    main()
