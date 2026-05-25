#!/usr/bin/env python3
"""Minimal VGGT inference service for dev-machine deployment."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

MODEL_ID = "facebook/VGGT-1B"
HOST = "0.0.0.0"
PORT = 18080

app = FastAPI(title="VGGT Inference Service", version="0.1.0")

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_USE_AUTOCAST = _DEVICE == "cuda"
if _USE_AUTOCAST and torch.cuda.get_device_capability()[0] >= 8:
    _AMP_DTYPE = torch.bfloat16
else:
    _AMP_DTYPE = torch.float16

print(f"[VGGT Service] Loading model {MODEL_ID} on {_DEVICE}...")
_MODEL = VGGT.from_pretrained(MODEL_ID).to(_DEVICE).eval()
print("[VGGT Service] Model loaded.")


def _tensor_shape(tensor: torch.Tensor) -> list[int]:
    return [int(v) for v in tensor.shape]


def _inference_summary(predictions: dict[str, torch.Tensor], image_h: int, image_w: int) -> dict[str, Any]:
    pose_enc = predictions["pose_enc"]
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, (image_h, image_w))

    depth = predictions["depth"]
    depth_conf = predictions["depth_conf"]
    world_points = predictions["world_points"]
    world_points_conf = predictions["world_points_conf"]

    return {
        "fields": [
            "pose_enc",
            "extrinsic",
            "intrinsic",
            "depth",
            "depth_conf",
            "world_points",
            "world_points_conf",
        ],
        "shapes": {
            "pose_enc": _tensor_shape(pose_enc),
            "extrinsic": _tensor_shape(extrinsic),
            "intrinsic": _tensor_shape(intrinsic),
            "depth": _tensor_shape(depth),
            "depth_conf": _tensor_shape(depth_conf),
            "world_points": _tensor_shape(world_points),
            "world_points_conf": _tensor_shape(world_points_conf),
        },
        "stats": {
            "depth_min": float(depth.min().item()),
            "depth_max": float(depth.max().item()),
            "depth_conf_mean": float(depth_conf.mean().item()),
            "world_points_abs_mean": float(world_points.abs().mean().item()),
            "world_points_conf_mean": float(world_points_conf.mean().item()),
        },
    }


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "device": _DEVICE, "model_id": MODEL_ID}


@app.post("/infer")
async def infer(images: list[UploadFile] = File(...)) -> dict[str, Any]:
    if not images:
        raise HTTPException(status_code=400, detail="At least one image is required")

    started_at = time.time()
    with tempfile.TemporaryDirectory(prefix="vggt-infer-") as tmpdir:
        local_paths: list[str] = []
        for idx, image in enumerate(images):
            suffix = Path(image.filename or "image.png").suffix or ".png"
            raw = await image.read()
            if not raw:
                raise HTTPException(status_code=400, detail=f"Image at index {idx} is empty")

            local_path = Path(tmpdir) / f"{idx:03d}{suffix}"
            local_path.write_bytes(raw)
            local_paths.append(str(local_path))

        model_input = load_and_preprocess_images(local_paths).to(_DEVICE)
        img_h, img_w = model_input.shape[-2], model_input.shape[-1]

        with torch.no_grad():
            if _USE_AUTOCAST:
                with torch.cuda.amp.autocast(dtype=_AMP_DTYPE):
                    predictions = _MODEL(model_input)
            else:
                predictions = _MODEL(model_input)

        elapsed_ms = (time.time() - started_at) * 1000.0
        return {
            "model_id": MODEL_ID,
            "device": _DEVICE,
            "num_input_images": len(local_paths),
            "preprocessed_image_size": [int(img_h), int(img_w)],
            "elapsed_ms": round(elapsed_ms, 2),
            "result": _inference_summary(predictions, img_h, img_w),
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("scripts.vggt_inference_service:app", host=HOST, port=PORT, log_level="info")
