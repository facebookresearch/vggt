# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

from __future__ import annotations
import os
import json
import glob
import time
import argparse
from collections import deque
from datetime import datetime
from typing import List, Tuple, Optional, Deque

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image  # for saving processed inputs

torch.set_float32_matmul_precision("high")
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

try:
    from torch.backends.cuda import sdp_kernel
    sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True)
except Exception:
    pass

# --- vggt imports ---
import sys
sys.path.append("vggt/")
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images   # AR-preserving like the demo
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------- helpers -----------------------------
def _mat_to_euler_xyz_deg(R: np.ndarray):
    sy = float(np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0]))
    singular = sy < 1e-6
    if not singular:
        roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:
        roll = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw = 0.0
    return float(roll), float(pitch), float(yaw)


def _cams_from_extri_intri(E: np.ndarray, K: np.ndarray, names: List[str]):
    #normalize to (S,3,4) and (S,3,3)
    if E.ndim == 4 and E.shape[0] == 1:
        E = E[0]
    if K.ndim == 4 and K.shape[0] == 1:
        K = K[0]
    if E.ndim != 3 or E.shape[-2:] != (3, 4):
        raise ValueError(f"Extrinsic must be (S,3,4); got {E.shape}")
    if K.ndim != 3 or K.shape[-2:] != (3, 3):
        raise ValueError(f"Intrinsic must be (S,3,3); got {K.shape}")

    cams = []
    S = E.shape[0]
    for i in range(S):
        Ei = E[i]  # (3,4) matrix
        H = np.eye(4, dtype=Ei.dtype); H[:3, :4] = Ei
        Twc = np.linalg.inv(H)
        Rwc, t = Twc[:3, :3], Twc[:3, 3]
        roll, pitch, yaw = _mat_to_euler_xyz_deg(Rwc)
        fx = float(K[i][0, 0]); fy = float(K[i][1, 1])
        cx = float(K[i][0, 2]); cy = float(K[i][1, 2])
        cams.append({
            "index": int(i),
            "image": os.path.basename(names[i]) if i < len(names) else f"{i:06d}.png",
            "position_m": {"x": float(t[0]), "y": float(t[1]), "z": float(t[2])},
            "euler_xyz_deg": {"roll": roll, "pitch": pitch, "yaw": yaw},
            "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
        })
    return cams


# --------------------------- main class ----------------------------
class CameraOnlyVGGT:
    def __init__(self, size: int = 384, window: int = 4, dtype: Optional[torch.dtype] = None, compile_model: bool = False):
        """
        Args:
          size: **long-side cap** in pixels (AR preserved). If images are smaller, no upscaling.
                For exact demo parity, pass a large cap (e.g., --size 4096) so we don't downscale.
          window: number of most-recent frames to feed per inference
          dtype: torch.bfloat16 (Ada+), torch.float16, or None to auto-pick
          compile_model: optional torch.compile; default False (Windows Dynamo can be flaky)
        """
        self.size = int(size)
        self.window = int(window)
        self.queue: Deque[str] = deque(maxlen=self.window)

        #cache of last processed inputs for saving
        self._last_images_cpu: Optional[torch.Tensor] = None  #(S,3,H,W) float32 [0,1] on CPU
        self._last_names: List[str] = []

        #camera head only
        self.model = VGGT(enable_camera=True, enable_point=False, enable_depth=False, enable_track=False)
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        sd = torch.hub.load_state_dict_from_url(_URL, map_location="cpu")
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        if unexpected or missing:
            print(f"[VGGT] Loaded weights with strict=False → missing={len(missing)}, unexpected={len(unexpected)}")
        self.model.eval().to(DEVICE)

        if dtype is None:
            if DEVICE == "cuda" and torch.cuda.get_device_capability()[0] >= 8:
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        else:
            self.dtype = dtype

        if compile_model:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=True)
            except Exception as e:
                print(f"[VGGT] torch.compile disabled (error: {e})")

        #lightweight warmup
        if DEVICE == "cuda":
            patch_h, patch_w = self._get_patch_hw()
            H = self._align_to_patch(min(max(self.size, patch_h), 2 * self.size), patch_h)
            W = self._align_to_patch(min(max(self.size, patch_w), 2 * self.size), patch_w)
            dummy = torch.zeros(2, 3, H, W, device=DEVICE)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(DEVICE == "cuda"), dtype=self.dtype):
                _ = self.model(dummy)
            torch.cuda.synchronize()

    # ---- internal: model patch helpers ----
    def _get_patch_hw(self) -> Tuple[int, int]:
        try:
            ps = getattr(self.model.aggregator.patch_embed, "patch_size", (16, 16))
        except Exception:
            ps = (16, 16)
        if isinstance(ps, (tuple, list)):
            return int(ps[0]), int(ps[1])
        return int(ps), int(ps)

    @staticmethod
    def _align_to_patch(n: int, patch: int) -> int:
        if patch <= 0:
            return int(n)
        return max(patch, (int(round(n / patch)) * patch))

    # ---------------- live-time helpers ----------------
    def push_path(self, img_path: str):
        self.queue.append(img_path)

    def _stack(self) -> Tuple[torch.Tensor, List[str]]:
        names = list(self.queue)
        if not names:
            raise ValueError("No frames in queue")

        out = load_and_preprocess_images(names)
        imgs = out[0] if isinstance(out, (tuple, list)) else out  # (S,3,H,W)

        #cap long side
        H, W = imgs.shape[-2:]
        patch_h, patch_w = self._get_patch_hw()

        if max(H, W) > self.size:
            scale = self.size / float(max(H, W))
            H2 = max(patch_h, int(round(H * scale)))
            W2 = max(patch_w, int(round(W * scale)))
            H2 = self._align_to_patch(H2, patch_h)
            W2 = self._align_to_patch(W2, patch_w)
            imgs = F.interpolate(imgs, size=(H2, W2), mode="bilinear", align_corners=False)

        return imgs.to(DEVICE), names

    @torch.no_grad()
    def infer_latest(self) -> dict:
        """Run VGGT on the current window; return latest frame's camera + timings."""
        if len(self.queue) == 0:
            raise ValueError("No frames queued")

        timings = {}
        t0 = time.time()

        t_load0 = time.time()
        images, names = self._stack()
        timings["load_preprocess_s"] = round(time.time() - t_load0, 4)

        #cache exactly what we compute on
        self._last_images_cpu = images.detach().to("cpu", dtype=torch.float32).clamp_(0.0, 1.0)
        self._last_names = list(names)

        t_inf0 = time.time()
        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda"), dtype=self.dtype):
            preds = self.model(images)
        timings["inference_s"] = round(time.time() - t_inf0, 4)

        H, W = images.shape[-2:]
        timings["proc_shape_hw"] = [int(H), int(W)]  #record processed size

        t_pose0 = time.time()
        extri, intri = pose_encoding_to_extri_intri(preds["pose_enc"], (H, W))
        timings["pose_decode_s"] = round(time.time() - t_pose0, 4)

        extri_np = extri.detach().cpu().numpy()
        intri_np = intri.detach().cpu().numpy()
        if extri_np.ndim == 4 and extri_np.shape[0] == 1:
            extri_np = extri_np[0]
        if intri_np.ndim == 4 and intri_np.shape[0] == 1:
            intri_np = intri_np[0]

        cams = _cams_from_extri_intri(extri_np, intri_np, names)

        timings["camera_extract_s"] = 0.0
        timings["num_frames"] = len(cams)
        timings["total_s"] = round(time.time() - t0, 4)

        latest = cams[-1]
        return {"cameras": cams, "latest": latest, "timings": timings}

    #---------------- saving helpers ----------------
    def save_processed_images(self, out_dir: str, format: str = "jpg", quality: int = 90) -> List[str]:
        """
        Save the *processed* inputs 
        that were used for the most recent inference.

        Args:
          out_dir: target directory, which is created if needed
          format: "jpg" or "png"
        Returns:
          List of saved file paths in order.
        """
        if self._last_images_cpu is None or not self._last_names:
            raise RuntimeError("No cached inputs to save; run infer_latest() first.")

        os.makedirs(out_dir, exist_ok=True)
        saved = []
        S, _, H, W = self._last_images_cpu.shape

        for i in range(S):
            #to uint8 HxWx3
            arr = (self._last_images_cpu[i].permute(1, 2, 0).numpy() * 255.0)
            arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

            stem = os.path.splitext(os.path.basename(self._last_names[i]))[0]
            ext = ".jpg" if format.lower() == "jpg" else ".png"
            path = os.path.join(out_dir, f"{i:04d}_{stem}{ext}")

            if format.lower() == "jpg":
                img.save(path, format="JPEG", quality=int(quality), optimize=True, subsampling=1)
            else:
                img.save(path, format="PNG", optimize=True)

            saved.append(path)

        return saved


# --------------------------- CLI / UI ------------------------------
def run_folder(images_dir: str, size: int, window: int, cap: int = 0):
    all_imgs = sorted(glob.glob(os.path.join(images_dir, "*")))
    if cap and cap > 0:
        all_imgs = all_imgs[:cap]
    if not all_imgs:
        raise SystemExit(f"No images found in {images_dir}")

    engine = CameraOnlyVGGT(size=size, window=window)

    for p in all_imgs:
        engine.push_path(p)
    out = engine.infer_latest()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    tgt = os.path.join(os.path.dirname(images_dir.rstrip(os.sep)), f"camera_only_{ts}")
    os.makedirs(tgt, exist_ok=True)

    #saving the JSONs
    with open(os.path.join(tgt, "cameras.json"), "w") as f:
        json.dump(out["cameras"], f, indent=2)
    with open(os.path.join(tgt, "timings.json"), "w") as f:
        json.dump(out["timings"], f, indent=2)

    #save the exact processed inputs used for compute
    inputs_dir = os.path.join(tgt, "compressed_inputs")
    saved_paths = engine.save_processed_images(inputs_dir, format="jpg", quality=90)

    #optional mapping for convenience
    mapping = [
        {"original": os.path.abspath(p), "saved": os.path.abspath(sp)}
        for p, sp in zip(all_imgs[-len(saved_paths):], saved_paths)
    ]
    with open(os.path.join(tgt, "inputs_mapping.json"), "w") as f:
        json.dump(mapping, f, indent=2)

    print("Latest frame camera:", json.dumps(out["latest"], indent=2))
    print("Timings:", json.dumps(out["timings"], indent=2))
    print(f"[saved] processed inputs → {inputs_dir}")

def launch_ui(size: int, window: int):
    import gradio as gr

    def on_upload(files):
        nonlocal_engine.queue.clear()
        for f in files:
            nonlocal_engine.push_path(f.name if hasattr(f, 'name') else f)
        res = nonlocal_engine.infer_latest()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_root = os.path.join(os.getcwd(), f"camera_only_ui_{ts}")
        os.makedirs(out_root, exist_ok=True)

        #saving the camera information along with the timings
        cams_path = os.path.join(out_root, "cameras.json")
        with open(cams_path, "w") as f:
            json.dump(res["cameras"], f, indent=2)
        with open(os.path.join(out_root, "timings.json"), "w") as f:
            json.dump(res["timings"], f, indent=2)

        #save processed inputs used
        inputs_dir = os.path.join(out_root, "compressed_inputs")
        nonlocal_engine.save_processed_images(inputs_dir, format="jpg", quality=90)

        log = (
            f"num_frames={res['timings']['num_frames']}, "
            f"proc_hw={res['timings'].get('proc_shape_hw')}, "
            f"load={res['timings']['load_preprocess_s']}s, "
            f"infer={res['timings']['inference_s']}s, "
            f"pose={res['timings']['pose_decode_s']}s, "
            f"total={res['timings']['total_s']}s\n"
            f"latest={json.dumps(res['latest'], indent=2)}\n"
            f"saved processed inputs → {inputs_dir}"
        )
        return log, cams_path

    nonlocal_engine = CameraOnlyVGGT(size=size, window=window)

    with gr.Blocks() as demo:
        gr.Markdown("no glb computations here, just the camera information")
        files = gr.File(file_count="multiple", label="Upload images (ordered)")
        log = gr.Markdown("Upload a few frames; we'll run a single pass on the window.")
        cams_file = gr.File(label="Download cameras.json")
        files.change(on_upload, inputs=[files], outputs=[log, cams_file])
    demo.queue(max_size=20).launch(show_error=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, default=None, help="Folder of images for a quick test")
    parser.add_argument("--size", type=int, default=384,
                        help="**Long-side cap** (px). AR preserved; set a large value for demo parity.")
    parser.add_argument("--window", type=int, default=4, help="Frames per inference (keep small for latency)")
    parser.add_argument("--cap", type=int, default=0, help="Use only first N images (0 = all)")
    parser.add_argument("--ui", action="store_true", help="Launch a minimal Gradio UI for manual timing")
    args = parser.parse_args()

    if args.ui:
        launch_ui(size=args.size, window=args.window)
    elif args.images_dir:
        run_folder(args.images_dir, size=args.size, window=args.window, cap=args.cap)
    else:
        print("Nothing to do. Pass --images_dir or --ui.")
