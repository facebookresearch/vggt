#!/usr/bin/env python3
"""Export VGGT core outputs (pose, depth, unprojected points) to ONNX."""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
from typing import Iterable, Tuple

import torch

from vggt.models.vggt import VGGT
if __package__ in {None, ""}:
    import sys

    THIS_DIR = Path(__file__).resolve().parent
    REPO_ROOT = THIS_DIR.parent.parent
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

class VGGTExportModule(torch.nn.Module):
    def __init__(self, model: VGGT, height: int, width: int, amp_dtype: torch.dtype | None) -> None:
        super().__init__()
        self.model = model
        self.amp_dtype = amp_dtype
        y, x = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing="ij",
        )
        self.register_buffer("grid_x", x)
        self.register_buffer("grid_y", y)
        self.height = height
        self.width = width

    @staticmethod
    def _quat_to_matrix(quat: torch.Tensor) -> torch.Tensor:
        q = quat / torch.clamp(quat.norm(dim=-1, keepdim=True), min=1e-8)
        x, y, z, w = torch.unbind(q, dim=-1)
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        m00 = 1 - 2 * (yy + zz)
        m01 = 2 * (xy - wz)
        m02 = 2 * (xz + wy)
        m10 = 2 * (xy + wz)
        m11 = 1 - 2 * (xx + zz)
        m12 = 2 * (yz - wx)
        m20 = 2 * (xz - wy)
        m21 = 2 * (yz + wx)
        m22 = 1 - 2 * (xx + yy)

        matrix = torch.stack(
            (m00, m01, m02, m10, m11, m12, m20, m21, m22), dim=-1
        ).reshape(quat.shape[:-1] + (3, 3))
        return matrix

    def _decode_pose(self, pose_encoding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        T = pose_encoding[..., :3]
        quat = pose_encoding[..., 3:7]
        fov_h = pose_encoding[..., 7]
        fov_w = pose_encoding[..., 8]

        R = self._quat_to_matrix(quat)
        extrinsics = torch.cat([R, T.unsqueeze(-1)], dim=-1)

        h = float(self.height)
        w = float(self.width)
        fy = (h / 2.0) / torch.tan(fov_h / 2.0)
        fx = (w / 2.0) / torch.tan(fov_w / 2.0)

        intrinsics = torch.zeros(pose_encoding.shape[:-1] + (3, 3), device=pose_encoding.device)
        intrinsics[..., 0, 0] = fx
        intrinsics[..., 1, 1] = fy
        intrinsics[..., 0, 2] = w / 2.0
        intrinsics[..., 1, 2] = h / 2.0
        intrinsics[..., 2, 2] = 1.0

        return extrinsics, intrinsics

    def _unproject(
        self,
        depth: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsics: torch.Tensor,
    ) -> torch.Tensor:
        depth = depth.squeeze(-1).float()
        B, S, H, W = depth.shape

        gx = self.grid_x.view(1, 1, H, W).expand(B, S, H, W)
        gy = self.grid_y.view(1, 1, H, W).expand(B, S, H, W)

        intrinsics = intrinsics.float()
        extrinsics = extrinsics.float()

        fx = intrinsics[..., 0, 0].unsqueeze(-1).unsqueeze(-1)
        fy = intrinsics[..., 1, 1].unsqueeze(-1).unsqueeze(-1)
        cx = intrinsics[..., 0, 2].unsqueeze(-1).unsqueeze(-1)
        cy = intrinsics[..., 1, 2].unsqueeze(-1).unsqueeze(-1)

        x_cam = (gx - cx) * depth / torch.clamp(fx, min=1e-6)
        y_cam = (gy - cy) * depth / torch.clamp(fy, min=1e-6)
        z_cam = depth
        cam_coords = torch.stack((x_cam, y_cam, z_cam), dim=-1)  # (B,S,H,W,3)

        R = extrinsics[..., :3, :3]
        t = extrinsics[..., :3, 3]
        R_t = R.transpose(-1, -2)

        cam_minus_t = cam_coords - t.unsqueeze(-2).unsqueeze(-2)
        world = torch.einsum("bshwd,bsde->bshwe", cam_minus_t, R_t)
        return world

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if images.ndim != 4:
            raise ValueError("Expected input images shape (views, 3, H, W)")

        images = images.unsqueeze(0)  # (1, S, 3, H, W)
        use_amp = self.amp_dtype is not None and images.is_cuda
        autocast_ctx = torch.cuda.amp.autocast(dtype=self.amp_dtype) if use_amp else contextlib.nullcontext()

        with torch.no_grad():
            with autocast_ctx:
                predictions = self.model(images)

        pose_enc = predictions["pose_enc"].float()
        depth_map = predictions["depth"].float()
        depth_conf = predictions.get("depth_conf", torch.ones_like(depth_map))

        extrinsics, intrinsics = self._decode_pose(pose_enc.float())
        world_points_unproj = self._unproject(depth_map, extrinsics, intrinsics)

        point_map = predictions.get("world_points")
        if point_map is None:
            point_map = world_points_unproj
        else:
            point_map = point_map.float()
        point_conf = predictions.get("world_points_conf")
        if point_conf is None:
            point_conf = torch.ones_like(depth_conf)
        else:
            point_conf = point_conf.float()

        outputs = (
            pose_enc.squeeze(0),
            extrinsics.squeeze(0),
            intrinsics.squeeze(0),
            depth_map.squeeze(0),
            depth_conf.float().squeeze(0),
            point_map.squeeze(0),
            point_conf.squeeze(0),
            world_points_unproj.squeeze(0),
        )
        return tuple(out.float().contiguous() for out in outputs)


def load_vggt(weights: Path | None, model_name: str, device: torch.device) -> VGGT:
    if weights is not None:
        model = VGGT()
        state = torch.load(weights, map_location="cpu")
        model.load_state_dict(state, strict=True)
    else:
        model = VGGT.from_pretrained(model_name)
    model.to(device)
    model.eval()
    model.float()
    return model


def export_onnx(
    model: VGGT,
    output: Path,
    num_cams: int,
    height: int,
    width: int,
    opset: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> None:
    module = VGGTExportModule(model, height, width, amp_dtype).to(device)
    module.eval()
    dummy = torch.rand(num_cams, 3, height, width, device=device)

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        module,
        dummy,
        output,
        input_names=["images"],
        output_names=[
            "pose_encoding",
            "extrinsics",
            "intrinsics",
            "depth_map",
            "depth_confidence",
            "point_map",
            "point_confidence",
            "unprojected_points",
        ],
        dynamic_axes={
            "images": {0: "views"},
            "pose_encoding": {0: "views"},
            "extrinsics": {0: "views"},
            "intrinsics": {0: "views"},
            "depth_map": {0: "views"},
            "depth_confidence": {0: "views"},
            "point_map": {0: "views"},
            "point_confidence": {0: "views"},
            "unprojected_points": {0: "views"},
        },
        opset_version=opset,
        do_constant_folding=False,
        dynamo=True
    )

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export VGGT to ONNX with pose/depth outputs.")
    parser.add_argument("--weights", type=Path, help="Path to VGGT state_dict (.pt)")
    parser.add_argument("--model-name", default="facebook/VGGT-1B", help="HuggingFace model ID")
    parser.add_argument("--output", type=Path, required=True, help="Destination ONNX file")
    parser.add_argument("--num-cams", type=int, default=6, help="Number of camera views")
    parser.add_argument("--height", type=int, default=518, help="Image height")
    parser.add_argument("--width", type=int, default=518, help="Image width")
    parser.add_argument("--opset", type=int, default=18, help="ONNX opset version")
    parser.add_argument("--device", default="cuda", help="Device for export (default: cuda)")
    parser.add_argument(
        "--amp",
        choices=["auto", "fp16", "bf16", "none"],
        default="auto",
        help="Autocast dtype to reduce GPU memory usage (default: auto)",
    )
    return parser


def select_amp_dtype(mode: str, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda" or mode == "none":
        return None
    if mode == "fp16":
        return torch.float16
    if mode == "bf16":
        return torch.bfloat16
    if mode == "auto":
        major, _ = torch.cuda.get_device_capability(device)
        return torch.bfloat16 if major >= 8 else torch.float16
    raise ValueError(f"Unknown amp mode: {mode}")


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = load_vggt(args.weights, args.model_name, device)
    amp_dtype = select_amp_dtype(args.amp, device)

    export_onnx(
        model,
        args.output,
        num_cams=int(args.num_cams),
        height=int(args.height),
        width=int(args.width),
        opset=int(args.opset),
        device=device,
        amp_dtype=amp_dtype,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
