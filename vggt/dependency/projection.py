# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from .distortion import apply_distortion


def img_from_cam_np(
    intrinsics: np.ndarray, points_cam: np.ndarray, extra_params: np.ndarray | None = None, default: float = 0.0
) -> np.ndarray:
    """
    Apply intrinsics (and optional radial distortion) to camera-space points.

    Args
    ----
    intrinsics  : (B,3,3) camera matrix K.
    points_cam  : (B,3,N) homogeneous camera coords  (x, y, z)ᵀ.
    extra_params: (B, N) or (B, k) distortion params (k = 1,2,4) or None.
    default     : value used for np.nan replacement.

    Returns
    -------
    points2D : (B,N,2) pixel coordinates.
    """
    # 1. perspective divide  ───────────────────────────────────────
    z = points_cam[:, 2:3, :]  # (B,1,N)
    points_cam_norm = points_cam / z  # (B,3,N)
    uv = points_cam_norm[:, :2, :]  # (B,2,N)

    # 2. optional distortion ──────────────────────────────────────
    if extra_params is not None:
        uu, vv = apply_distortion(extra_params, uv[:, 0], uv[:, 1])
        uv = np.stack([uu, vv], axis=1)  # (B,2,N)

    # 3. homogeneous coords then K multiplication ─────────────────
    ones = np.ones_like(uv[:, :1, :])  # (B,1,N)
    points_cam_h = np.concatenate([uv, ones], axis=1)  # (B,3,N)

    # batched mat-mul: K · [u v 1]ᵀ
    points2D_h = np.einsum("bij,bjk->bik", intrinsics, points_cam_h)  # (B,3,N)
    points2D = np.nan_to_num(points2D_h[:, :2, :], nan=default)  # (B,2,N)

    return points2D.transpose(0, 2, 1)  # (B,N,2)


def project_3D_points_np(
    points3D: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray | None = None,
    extra_params: np.ndarray | None = None,
    *,
    default: float = 0.0,
    only_points_cam: bool = False,
):
    """
    NumPy clone of ``project_3D_points``.

    Parameters
    ----------
    points3D          : (N,3) world-space points.
    extrinsics        : (B,3,4)  [R|t] matrix for each of B cameras.
    intrinsics        : (B,3,3)  K matrix (optional if you only need cam-space).
    extra_params      : (B,k) or (B,N) distortion parameters (k ∈ {1,2,4}) or None.
    default           : value used to replace NaNs.
    only_points_cam   : if True, skip the projection and return points_cam with points2D as None.

    Returns
    -------
    (points2D, points_cam) : A tuple where points2D is (B,N,2) pixel coords or None if only_points_cam=True,
                           and points_cam is (B,3,N) camera-space coordinates.
    """
    # ----- 0. prep sizes -----------------------------------------------------
    N = points3D.shape[0]  # #points
    B = extrinsics.shape[0]  # #cameras

    # ----- 1. world → homogeneous -------------------------------------------
    w_h = np.ones((N, 1), dtype=points3D.dtype)
    points3D_h = np.concatenate([points3D, w_h], axis=1)  # (N,4)

    # broadcast to every camera (no actual copying with np.broadcast_to) ------
    points3D_h_B = np.broadcast_to(points3D_h, (B, N, 4))  # (B,N,4)

    # ----- 2. apply extrinsics  (camera frame) ------------------------------
    # X_cam = E · X_hom
    # einsum:  E_(b i j)  ·  X_(b n j)  →  (b n i)
    points_cam = np.einsum("bij,bnj->bni", extrinsics, points3D_h_B)  # (B,N,3)
    points_cam = points_cam.transpose(0, 2, 1)  # (B,3,N)

    if only_points_cam:
        return None, points_cam

    # ----- 3. intrinsics + distortion ---------------------------------------
    if intrinsics is None:
        raise ValueError("`intrinsics` must be provided unless only_points_cam=True")

    points2D = img_from_cam_np(intrinsics, points_cam, extra_params=extra_params, default=default)

    return points2D, points_cam


def project_3D_points(points3D, extrinsics, intrinsics=None, extra_params=None, default=0, only_points_cam=False):
    """
    Transforms 3D points to 2D using extrinsic and intrinsic parameters.
    Args:
        points3D (torch.Tensor): 3D points of shape Px3.
        extrinsics (torch.Tensor): Extrinsic parameters of shape Bx3x4.
        intrinsics (torch.Tensor): Intrinsic parameters of shape Bx3x3.
        extra_params (torch.Tensor): Extra parameters of shape BxN, used for radial distortion.
        default (float): Default value to replace NaNs.
        only_points_cam (bool): If True, skip the projection and return points2D as None.

    Returns:
        tuple: (points2D, points_cam) where points2D is of shape BxNx2 or None if only_points_cam=True,
               and points_cam is of shape Bx3xN.
    """
    with torch.cuda.amp.autocast(dtype=torch.double):
        N = points3D.shape[0]  # Number of points
        B = extrinsics.shape[0]  # Batch size, i.e., number of cameras
        points3D_homogeneous = torch.cat([points3D, torch.ones_like(points3D[..., 0:1])], dim=1)  # Nx4
        # Reshape for batch processing
        points3D_homogeneous = points3D_homogeneous.unsqueeze(0).expand(B, -1, -1)  # BxNx4

        # Step 1: Apply extrinsic parameters
        # Transform 3D points to camera coordinate system for all cameras
        points_cam = torch.bmm(extrinsics, points3D_homogeneous.transpose(-1, -2))

        if only_points_cam:
            return None, points_cam

        # Step 2: Apply intrinsic parameters and (optional) distortion
        points2D = img_from_cam(intrinsics, points_cam, extra_params, default)

        return points2D, points_cam


def img_from_cam(intrinsics, points_cam, extra_params=None, default=0.0):
    """
    Applies intrinsic parameters and optional distortion to the given 3D points.

    Args:
        intrinsics (torch.Tensor): Intrinsic camera parameters of shape Bx3x3.
        points_cam (torch.Tensor): 3D points in camera coordinates of shape Bx3xN.
        extra_params (torch.Tensor, optional): Distortion parameters of shape BxN, where N can be 1, 2, or 4.
        default (float, optional): Default value to replace NaNs in the output.

    Returns:
        points2D (torch.Tensor): 2D points in pixel coordinates of shape BxNx2.
    """

    # Normalize by the third coordinate (homogeneous division)
    points_cam = points_cam / points_cam[:, 2:3, :]
    # Extract uv
    uv = points_cam[:, :2, :]

    # Apply distortion if extra_params are provided
    if extra_params is not None:
        uu, vv = apply_distortion(extra_params, uv[:, 0], uv[:, 1])
        uv = torch.stack([uu, vv], dim=1)

    # Prepare points_cam for batch matrix multiplication
    points_cam_homo = torch.cat((uv, torch.ones_like(uv[:, :1, :])), dim=1)  # Bx3xN
    # Apply intrinsic parameters using batch matrix multiplication
    points2D_homo = torch.bmm(intrinsics, points_cam_homo)  # Bx3xN

    # Extract x and y coordinates
    points2D = points2D_homo[:, :2, :]  # Bx2xN

    # Replace NaNs with default value
    points2D = torch.nan_to_num(points2D, nan=default)

    return points2D.transpose(1, 2)  # BxNx2


if __name__ == "__main__":
    # Set up example input
    B, N = 24, 10240

    for _ in range(100):
        points3D = np.random.rand(N, 3).astype(np.float64)
        extrinsics = np.random.rand(B, 3, 4).astype(np.float64)
        intrinsics = np.random.rand(B, 3, 3).astype(np.float64)

        # Convert to torch tensors
        points3D_torch = torch.tensor(points3D)
        extrinsics_torch = torch.tensor(extrinsics)
        intrinsics_torch = torch.tensor(intrinsics)

        # Run NumPy implementation
        points2D_np, points_cam_np = project_3D_points_np(points3D, extrinsics, intrinsics)

        # Run torch implementation
        points2D_torch, points_cam_torch = project_3D_points(points3D_torch, extrinsics_torch, intrinsics_torch)

        # Convert torch output to numpy
        points2D_torch_np = points2D_torch.detach().numpy()
        points_cam_torch_np = points_cam_torch.detach().numpy()

        # Compute difference
        diff = np.abs(points2D_np - points2D_torch_np)
        print("Difference between NumPy and PyTorch implementations:")
        print(diff)

        # Check max error
        max_diff = np.max(diff)
        print(f"Maximum difference: {max_diff}")

        if np.allclose(points2D_np, points2D_torch_np, atol=1e-6):
            print("Implementations match closely.")
        else:
            print("Significant differences detected.")

        if points_cam_np is not None:
            points_cam_diff = np.abs(points_cam_np - points_cam_torch_np)
            print("Difference between NumPy and PyTorch camera-space coordinates:")
            print(points_cam_diff)

            # Check max error
            max_cam_diff = np.max(points_cam_diff)
            print(f"Maximum camera-space coordinate difference: {max_cam_diff}")

            if np.allclose(points_cam_np, points_cam_torch_np, atol=1e-6):
                print("Camera-space coordinates match closely.")
            else:
                print("Significant differences detected in camera-space coordinates.")
