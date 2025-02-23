# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from ...models.builder import LOSSES


@LOSSES.register_module()
class GradientLoss(nn.Module):
    """GradientLoss.

    Adapted from https://www.cs.cornell.edu/projects/megadepth/

    Args:
        valid_mask (bool): Whether filter invalid gt (gt > 0). Default: True.
        loss_weight (float): Weight of the loss. Default: 1.0.
        max_depth (int): When filtering invalid gt, set a max threshold. Default: None.
    """

    def __init__(self, valid_mask=True, loss_weight=1.0, max_depth=None, loss_name="loss_grad"):
        super(GradientLoss, self).__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.max_depth = max_depth
        self.loss_name = loss_name

        self.eps = 0.001  # avoid grad explode

    def gradientloss(self, input, target):
        input_downscaled = [input] + [input[:: 2 * i, :: 2 * i] for i in range(1, 4)]
        target_downscaled = [target] + [target[:: 2 * i, :: 2 * i] for i in range(1, 4)]

        gradient_loss = 0
        for input, target in zip(input_downscaled, target_downscaled):
            if self.valid_mask:
                mask = target > 0
                if self.max_depth is not None:
                    mask = torch.logical_and(target > 0, target <= self.max_depth)
                N = torch.sum(mask)
            else:
                mask = torch.ones_like(target)
                N = input.numel()
            input_log = torch.log(input + self.eps)
            target_log = torch.log(target + self.eps)
            log_d_diff = input_log - target_log

            log_d_diff = torch.mul(log_d_diff, mask)

            v_gradient = torch.abs(log_d_diff[0:-2, :] - log_d_diff[2:, :])
            v_mask = torch.mul(mask[0:-2, :], mask[2:, :])
            v_gradient = torch.mul(v_gradient, v_mask)

            h_gradient = torch.abs(log_d_diff[:, 0:-2] - log_d_diff[:, 2:])
            h_mask = torch.mul(mask[:, 0:-2], mask[:, 2:])
            h_gradient = torch.mul(h_gradient, h_mask)

            gradient_loss += (torch.sum(h_gradient) + torch.sum(v_gradient)) / N

        return gradient_loss

    def forward(self, depth_pred, depth_gt):
        """Forward function."""

        gradient_loss = self.loss_weight * self.gradientloss(depth_pred, depth_gt)
        return gradient_loss
