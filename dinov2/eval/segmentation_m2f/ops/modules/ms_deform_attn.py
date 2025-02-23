# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import math
import warnings

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function
from torch.cuda.amp import custom_fwd
from torch.nn.init import constant_, xavier_uniform_


class MSDeformAttnFunction(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(
        ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step
    ):
        output = ms_deform_attn_core_pytorch(
            value,
            value_spatial_shapes,
            #  value_level_start_index,
            sampling_locations,
            attention_weights,
        )
        return output


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations, attention_weights):
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights).sum(-1).view(N_, M_ * D_, Lq_)
    return output.transpose(1, 2).contiguous()


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n - 1) == 0) and n != 0


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, ratio=1.0):
        """Multi-Scale Deformable Attention Module.

        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads, " "but got {} and {}".format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2
        # which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn(
                "You'd better set d_model in MSDeformAttn to make "
                "the dimension of each attention head a power of 2 "
                "which is more efficient in our CUDA implementation."
            )

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.ratio = ratio
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, int(d_model * ratio))
        self.output_proj = nn.Linear(int(d_model * ratio), d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (
            (grid_init / grid_init.abs().max(-1, keepdim=True)[0])
            .view(self.n_heads, 1, 1, 2)
            .repeat(1, self.n_levels, self.n_points, 1)
        )
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1

        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)

    def forward(
        self,
        query,
        reference_points,
        input_flatten,
        input_spatial_shapes,
        input_level_start_index,
        input_padding_mask=None,
    ):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \\sum_{l=0}^{L-1} H_l \\cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \\sum_{l=0}^{L-1} H_l \\cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        # print(query.shape)
        # print(reference_points.shape)
        # print(input_flatten.shape)
        # print(input_spatial_shapes.shape)
        # print(input_level_start_index.shape)
        # print(input_spatial_shapes)
        # print(input_level_start_index)

        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        value = value.view(N, Len_in, self.n_heads, int(self.ratio * self.d_model) // self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(
                "Last dim of reference_points must be 2 or 4, but get {} instead.".format(reference_points.shape[-1])
            )
        output = MSDeformAttnFunction.apply(
            value,
            input_spatial_shapes,
            input_level_start_index,
            sampling_locations,
            attention_weights,
            self.im2col_step,
        )
        output = self.output_proj(output)
        return output
