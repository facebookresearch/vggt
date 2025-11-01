"""
GPU-first point cloud helpers for the reconstruction pipeline.

These utilities mirror the behaviour of the original NumPy versions but keep
the heavy lifting on torch tensors so we can minimise host/device transfers.
"""
from __future__ import annotations

from typing import Tuple

import torch


def unproject_depth_map_to_point_map(
    depth_map: torch.Tensor,
    extrinsics_cam: torch.Tensor,
    intrinsics_cam: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unproject a batch of depth maps to 3D world coordinates on the GPU.

    Args:
        depth_map: Tensor of shape (S, H, W) or (S, H, W, 1).
        extrinsics_cam: Tensor of shape (S, 3, 4) containing camera-from-world transforms.
        intrinsics_cam: Tensor of shape (S, 3, 3).
        eps: Minimum depth considered valid.

    Returns:
        point_map: Tensor of shape (S, H, W, 3) with XYZ world coordinates.
        valid_mask: Tensor of shape (S, H, W) indicating where depth > eps.
    """

    if depth_map.dim() == 4 and depth_map.shape[-1] == 1:
        depth_map = depth_map.squeeze(-1)
    if depth_map.dim() == 4 and depth_map.shape[1] == 1:
        depth_map = depth_map.squeeze(1)
    if depth_map.dim() != 3:
        raise ValueError(f"depth_map must be (S,H,W); got {tuple(depth_map.shape)}")

    depth_map = depth_map.float()
    extrinsics_cam = extrinsics_cam.float()
    intrinsics_cam = intrinsics_cam.float()

    device = depth_map.device
    dtype = depth_map.dtype

    batch, height, width = depth_map.shape
    valid_mask = depth_map > eps

    ys = torch.arange(height, device=device, dtype=dtype)
    xs = torch.arange(width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_y = grid_y.unsqueeze(0).expand(batch, -1, -1)
    grid_x = grid_x.unsqueeze(0).expand(batch, -1, -1)

    fx = intrinsics_cam[..., 0, 0].view(batch, 1, 1)
    fy = intrinsics_cam[..., 1, 1].view(batch, 1, 1)
    cx = intrinsics_cam[..., 0, 2].view(batch, 1, 1)
    cy = intrinsics_cam[..., 1, 2].view(batch, 1, 1)

    z_cam = depth_map
    x_cam = (grid_x - cx) / fx * z_cam
    y_cam = (grid_y - cy) / fy * z_cam
    cam_coords = torch.stack((x_cam, y_cam, z_cam), dim=-1)  # (S,H,W,3)

    R = extrinsics_cam[..., :3]
    t = extrinsics_cam[..., 3]
    R_trans = R.transpose(-1, -2)  # camera->world rotation
    t_world = (-torch.bmm(R_trans, t.unsqueeze(-1))).squeeze(-1)

    cam_flat = cam_coords.view(batch, -1, 3)
    world_flat = torch.bmm(cam_flat, R_trans) + t_world.unsqueeze(1)
    world_points = world_flat.view(batch, height, width, 3)

    return world_points, valid_mask

