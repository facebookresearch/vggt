# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modified from https://github.com/facebookresearch/PoseDiffusion
# and https://github.com/facebookresearch/co-tracker/tree/main


import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple, Union
from einops import rearrange, repeat


# from minipytorch3d.harmonic_embedding import HarmonicEmbedding

# from minipytorch3d.cameras import CamerasBase, PerspectiveCameras
# from minipytorch3d.rotation_conversions import (
#     matrix_to_quaternion,
#     quaternion_to_matrix,
# )


# from pytorch3d.renderer import HarmonicEmbedding
# from pytorch3d.renderer.cameras import CamerasBase, PerspectiveCameras
# from pytorch3d.transforms.rotation_conversions import matrix_to_quaternion, quaternion_to_matrix

# from ..utils.metric import closed_form_inverse_OpenCV
# from ..utils.triangulation import create_intri_matrix

EPS = 1e-9


# def get_EFP(pred_cameras, image_size, B, S, default_focal=False):
#     """
#     Converting PyTorch3D cameras to extrinsics, intrinsics matrix

#     Return extrinsics, intrinsics, focal_length, principal_point
#     """
#     scale = image_size.min()

#     focal_length = pred_cameras.focal_length

#     principal_point = torch.zeros_like(focal_length)

#     focal_length = focal_length * scale / 2
#     principal_point = (image_size[None] - principal_point * scale) / 2

#     Rots = pred_cameras.R.clone()
#     Trans = pred_cameras.T.clone()

#     extrinsics = torch.cat([Rots, Trans[..., None]], dim=-1)

#     # reshape
#     extrinsics = extrinsics.reshape(B, S, 3, 4)
#     focal_length = focal_length.reshape(B, S, 2)
#     principal_point = principal_point.reshape(B, S, 2)

#     # only one dof focal length
#     if default_focal:
#         focal_length[:] = scale
#     else:
#         focal_length = focal_length.mean(dim=-1, keepdim=True).expand(-1, -1, 2)
#         focal_length = focal_length.clamp(0.2 * scale, 5 * scale)

#     intrinsics = create_intri_matrix(focal_length, principal_point)

#     return extrinsics, intrinsics


# def pose_encoding_to_camera(
#     pose_encoding,
#     pose_encoding_type="absT_quaR_logFL",
#     log_focal_length_bias=1.8,
#     min_focal_length=0.1,
#     max_focal_length=30,
#     return_dict=False,
#     to_OpenCV=True,
# ):
#     """
#     Args:
#         pose_encoding: A tensor of shape `BxNxC`, containing a batch of
#                         `BxN` `C`-dimensional pose encodings.
#         pose_encoding_type: The type of pose encoding,
#     """
#     pose_encoding_reshaped = pose_encoding.reshape(
#         -1, pose_encoding.shape[-1]
#     )  # Reshape to BNxC

#     if pose_encoding_type == "absT_quaR_logFL":
#         # 3 for absT, 4 for quaR, 2 for absFL
#         abs_T = pose_encoding_reshaped[:, :3]
#         quaternion_R = pose_encoding_reshaped[:, 3:7]
#         R = quaternion_to_matrix(quaternion_R)
#         log_focal_length = pose_encoding_reshaped[:, 7:9]
#         # log_focal_length_bias was the hyperparameter
#         # to ensure the mean of logFL close to 0 during training
#         # Now converted back
#         focal_length = (log_focal_length + log_focal_length_bias).exp()
#         # clamp to avoid weird fl values
#         focal_length = torch.clamp(
#             focal_length, min=min_focal_length, max=max_focal_length
#         )
#     elif pose_encoding_type == "absT_quaR_OneFL":
#         # 3 for absT, 4 for quaR, 1 for absFL
#         # [absolute translation, quaternion rotation, normalized focal length]
#         abs_T = pose_encoding_reshaped[:, :3]
#         quaternion_R = pose_encoding_reshaped[:, 3:7]
#         R = quaternion_to_matrix(quaternion_R)
#         focal_length = pose_encoding_reshaped[:, 7:8]
#         focal_length = torch.clamp(
#             focal_length, min=min_focal_length, max=max_focal_length
#         )
#     else:
#         raise ValueError(f"Unknown pose encoding {pose_encoding_type}")

#     if to_OpenCV:
#         ### From Pytorch3D coordinate to OpenCV coordinate:
#         # I hate coordinate conversion
#         R = R.clone()
#         abs_T = abs_T.clone()
#         R[:, :, :2] *= -1
#         abs_T[:, :2] *= -1
#         R = R.permute(0, 2, 1)

#         extrinsics_4x4 = torch.eye(4, 4).to(R.dtype).to(R.device)
#         extrinsics_4x4 = extrinsics_4x4[None].repeat(len(R), 1, 1)

#         extrinsics_4x4[:, :3, :3] = R.clone()
#         extrinsics_4x4[:, :3, 3] = abs_T.clone()

#         rel_transform = closed_form_inverse_OpenCV(extrinsics_4x4[0:1])
#         rel_transform = rel_transform.expand(len(extrinsics_4x4), -1, -1)

#         # relative to the first camera
#         # NOTE it is extrinsics_4x4 x rel_transform instead of rel_transform x extrinsics_4x4
#         # this is different in opencv / pytorch3d convention
#         extrinsics_4x4 = torch.bmm(extrinsics_4x4, rel_transform)

#         R = extrinsics_4x4[:, :3, :3].clone()
#         abs_T = extrinsics_4x4[:, :3, 3].clone()

#     if return_dict:
#         return {"focal_length": focal_length, "R": R, "T": abs_T}

#     pred_cameras = PerspectiveCameras(
#         focal_length=focal_length, R=R, T=abs_T, device=R.device
#     )
#     return pred_cameras


# def camera_to_pose_encoding(
#     camera,
#     pose_encoding_type="absT_quaR_logFL",
#     log_focal_length_bias=1.8,
#     min_focal_length=0.1,
#     max_focal_length=30,
# ):
#     """
#     Inverse to pose_encoding_to_camera
#     """
#     if pose_encoding_type == "absT_quaR_logFL":
#         # Convert rotation matrix to quaternion
#         quaternion_R = matrix_to_quaternion(camera.R)

#         # Calculate log_focal_length
#         log_focal_length = (
#             torch.log(
#                 torch.clamp(
#                     camera.focal_length,
#                     min=min_focal_length,
#                     max=max_focal_length,
#                 )
#             )
#             - log_focal_length_bias
#         )

#         # Concatenate to form pose_encoding
#         pose_encoding = torch.cat(
#             [camera.T, quaternion_R, log_focal_length], dim=-1
#         )

#     elif pose_encoding_type == "absT_quaR_OneFL":
#         # [absolute translation, quaternion rotation, normalized focal length]
#         quaternion_R = matrix_to_quaternion(camera.R)
#         focal_length = (
#             torch.clamp(
#                 camera.focal_length, min=min_focal_length, max=max_focal_length
#             )
#         )[..., 0:1]
#         pose_encoding = torch.cat(
#             [camera.T, quaternion_R, focal_length], dim=-1
#         )
#     else:
#         raise ValueError(f"Unknown pose encoding {pose_encoding_type}")

#     return pose_encoding


# class PoseEmbedding(nn.Module):
#     def __init__(self, target_dim, n_harmonic_functions=10, append_input=True):
#         super().__init__()

#         self._emb_pose = HarmonicEmbedding(
#             n_harmonic_functions=n_harmonic_functions, append_input=append_input
#         )

#         self.out_dim = self._emb_pose.get_output_dim(target_dim)

#     def forward(self, pose_encoding):
#         e_pose_encoding = self._emb_pose(pose_encoding)
#         return e_pose_encoding


def get_2d_sincos_pos_embed(
    embed_dim: int, grid_size: Union[int, Tuple[int, int]], return_grid=False
) -> torch.Tensor:
    """
    This function initializes a grid and generates a 2D positional embedding using sine and cosine functions.
    It is a wrapper of get_2d_sincos_pos_embed_from_grid.
    Args:
    - embed_dim: The embedding dimension.
    - grid_size: The grid size.
    Returns:
    - pos_embed: The generated 2D positional embedding.
    """
    if isinstance(grid_size, tuple):
        grid_size_h, grid_size_w = grid_size
    else:
        grid_size_h = grid_size_w = grid_size
    grid_h = torch.arange(grid_size_h, dtype=torch.float)
    grid_w = torch.arange(grid_size_w, dtype=torch.float)
    grid = torch.meshgrid(grid_w, grid_h, indexing="xy")
    grid = torch.stack(grid, dim=0)
    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if return_grid:
        return (
            pos_embed.reshape(1, grid_size_h, grid_size_w, -1).permute(
                0, 3, 1, 2
            ),
            grid,
        )
    return pos_embed.reshape(1, grid_size_h, grid_size_w, -1).permute(
        0, 3, 1, 2
    )


def get_2d_sincos_pos_embed_from_grid(
    embed_dim: int, grid: torch.Tensor
) -> torch.Tensor:
    """
    This function generates a 2D positional embedding from a given grid using sine and cosine functions.

    Args:
    - embed_dim: The embedding dimension.
    - grid: The grid to generate the embedding from.

    Returns:
    - emb: The generated 2D positional embedding.
    """
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[0]
    )  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(
        embed_dim // 2, grid[1]
    )  # (H*W, D/2)

    emb = torch.cat([emb_h, emb_w], dim=2)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(
    embed_dim: int, pos: torch.Tensor
) -> torch.Tensor:
    """
    This function generates a 1D positional embedding from a given grid using sine and cosine functions.

    Args:
    - embed_dim: The embedding dimension.
    - pos: The position to generate the embedding from.

    Returns:
    - emb: The generated 1D positional embedding.
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.double)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb[None].float()


def get_2d_embedding(
    xy: torch.Tensor, C: int, cat_coords: bool = True
) -> torch.Tensor:
    """
    This function generates a 2D positional embedding from given coordinates using sine and cosine functions.

    Args:
    - xy: The coordinates to generate the embedding from.
    - C: The size of the embedding.
    - cat_coords: A flag to indicate whether to concatenate the original coordinates to the embedding.

    Returns:
    - pe: The generated 2D positional embedding.
    """
    B, N, D = xy.shape
    assert D == 2

    x = xy[:, :, 0:1]
    y = xy[:, :, 1:2]
    div_term = (
        torch.arange(0, C, 2, device=xy.device, dtype=torch.float32)
        * (1000.0 / C)
    ).reshape(1, 1, int(C / 2))

    pe_x = torch.zeros(B, N, C, device=xy.device, dtype=torch.float32)
    pe_y = torch.zeros(B, N, C, device=xy.device, dtype=torch.float32)

    pe_x[:, :, 0::2] = torch.sin(x * div_term)
    pe_x[:, :, 1::2] = torch.cos(x * div_term)

    pe_y[:, :, 0::2] = torch.sin(y * div_term)
    pe_y[:, :, 1::2] = torch.cos(y * div_term)

    pe = torch.cat([pe_x, pe_y], dim=2)  # (B, N, C*3)
    if cat_coords:
        pe = torch.cat([xy, pe], dim=2)  # (B, N, C*3+3)
    return pe


def bilinear_sampler(input, coords, align_corners=True, padding_mode="border"):
    r"""Sample a tensor using bilinear interpolation

    `bilinear_sampler(input, coords)` samples a tensor :attr:`input` at
    coordinates :attr:`coords` using bilinear interpolation. It is the same
    as `torch.nn.functional.grid_sample()` but with a different coordinate
    convention.

    The input tensor is assumed to be of shape :math:`(B, C, H, W)`, where
    :math:`B` is the batch size, :math:`C` is the number of channels,
    :math:`H` is the height of the image, and :math:`W` is the width of the
    image. The tensor :attr:`coords` of shape :math:`(B, H_o, W_o, 2)` is
    interpreted as an array of 2D point coordinates :math:`(x_i,y_i)`.

    Alternatively, the input tensor can be of size :math:`(B, C, T, H, W)`,
    in which case sample points are triplets :math:`(t_i,x_i,y_i)`. Note
    that in this case the order of the components is slightly different
    from `grid_sample()`, which would expect :math:`(x_i,y_i,t_i)`.

    If `align_corners` is `True`, the coordinate :math:`x` is assumed to be
    in the range :math:`[0,W-1]`, with 0 corresponding to the center of the
    left-most image pixel :math:`W-1` to the center of the right-most
    pixel.

    If `align_corners` is `False`, the coordinate :math:`x` is assumed to
    be in the range :math:`[0,W]`, with 0 corresponding to the left edge of
    the left-most pixel :math:`W` to the right edge of the right-most
    pixel.

    Similar conventions apply to the :math:`y` for the range
    :math:`[0,H-1]` and :math:`[0,H]` and to :math:`t` for the range
    :math:`[0,T-1]` and :math:`[0,T]`.

    Args:
        input (Tensor): batch of input images.
        coords (Tensor): batch of coordinates.
        align_corners (bool, optional): Coordinate convention. Defaults to `True`.
        padding_mode (str, optional): Padding mode. Defaults to `"border"`.

    Returns:
        Tensor: sampled points.
    """
    coords = coords.detach().clone()
    ############################################################
    # IMPORTANT:
    coords = coords.to(input.device).to(input.dtype)
    ############################################################

    sizes = input.shape[2:]

    assert len(sizes) in [2, 3]

    if len(sizes) == 3:
        # t x y -> x y t to match dimensions T H W in grid_sample
        coords = coords[..., [1, 2, 0]]

    if align_corners:
        # coords = coords * torch.tensor(
        #     [2 / max(size - 1, 1) for size in reversed(sizes)],
        #     device=coords.device,
        # )
        scale = torch.tensor([2 / max(size - 1, 1) for size in reversed(sizes)],
                        device=coords.device, dtype=coords.dtype)
    else:
        # coords = coords * torch.tensor(
        #     [2 / size for size in reversed(sizes)], device=coords.device
        # )
        scale = torch.tensor([2 / size for size in reversed(sizes)],
                             device=coords.device, dtype=coords.dtype)


    coords.mul_(scale)  # coords = coords * scale
    coords.sub_(1)      # coords = coords - 1

    return F.grid_sample(
        input, coords, align_corners=align_corners, padding_mode=padding_mode
    )


def sample_features4d(input, coords):
    r"""Sample spatial features

    `sample_features4d(input, coords)` samples the spatial features
    :attr:`input` represented by a 4D tensor :math:`(B, C, H, W)`.

    The field is sampled at coordinates :attr:`coords` using bilinear
    interpolation. :attr:`coords` is assumed to be of shape :math:`(B, R,
    2)`, where each sample has the format :math:`(x_i, y_i)`. This uses the
    same convention as :func:`bilinear_sampler` with `align_corners=True`.

    The output tensor has one feature per point, and has shape :math:`(B,
    R, C)`.

    Args:
        input (Tensor): spatial features.
        coords (Tensor): points.

    Returns:
        Tensor: sampled features.
    """

    B, _, _, _ = input.shape

    # B R 2 -> B R 1 2
    coords = coords.unsqueeze(2)

    # B C R 1
    feats = bilinear_sampler(input, coords)

    return feats.permute(0, 2, 1, 3).view(
        B, -1, feats.shape[1] * feats.shape[3]
    )  # B C R 1 -> B R C



