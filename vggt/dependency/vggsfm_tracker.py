# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

from hydra.utils import instantiate
from omegaconf import OmegaConf

from .track_modules.track_refine import refine_track
from .track_modules.blocks import BasicEncoder, ShallowEncoder
from .track_modules.base_track_predictor import BaseTrackerPredictor


class TrackerPredictor(nn.Module):
    def __init__(self, **extra_args):
        super(TrackerPredictor, self).__init__()
        """
        Initializes the tracker predictor.

        Both coarse_predictor and fine_predictor are constructed as a BaseTrackerPredictor,
        check track_modules/base_track_predictor.py

        Both coarse_fnet and fine_fnet are constructed as a 2D CNN network
        check track_modules/blocks.py for BasicEncoder and ShallowEncoder
        """
        # Define coarse predictor configuration
        coarse_stride = 4
        self.coarse_down_ratio = 2

        # Create networks directly instead of using instantiate
        self.coarse_fnet = BasicEncoder(stride=coarse_stride)
        self.coarse_predictor = BaseTrackerPredictor(stride=coarse_stride)

        # Create fine predictor with stride = 1
        self.fine_fnet = ShallowEncoder(stride=1)
        self.fine_predictor = BaseTrackerPredictor(
            stride=1,
            depth=4,
            corr_levels=3,
            corr_radius=3,
            latent_dim=32,
            hidden_size=256,
            fine=True,
            use_spaceatt=False,
        )

    def forward(
        self, images, query_points, fmaps=None, coarse_iters=6, inference=True, fine_tracking=True, fine_chunk=40960
    ):
        """
        Args:
            images (torch.Tensor): Images as RGB, in the range of [0, 1], with a shape of B x S x 3 x H x W.
            query_points (torch.Tensor): 2D xy of query points, relative to top left, with a shape of B x N x 2.
            fmaps (torch.Tensor, optional): Precomputed feature maps. Defaults to None.
            coarse_iters (int, optional): Number of iterations for coarse prediction. Defaults to 6.
            inference (bool, optional): Whether to perform inference. Defaults to True.
            fine_tracking (bool, optional): Whether to perform fine tracking. Defaults to True.

        Returns:
            tuple: A tuple containing fine_pred_track, coarse_pred_track, pred_vis, and pred_score.
        """

        if fmaps is None:
            batch_num, frame_num, image_dim, height, width = images.shape
            reshaped_image = images.reshape(batch_num * frame_num, image_dim, height, width)
            fmaps = self.process_images_to_fmaps(reshaped_image)
            fmaps = fmaps.reshape(batch_num, frame_num, -1, fmaps.shape[-2], fmaps.shape[-1])

            if inference:
                torch.cuda.empty_cache()

        # Coarse prediction
        coarse_pred_track_lists, pred_vis = self.coarse_predictor(
            query_points=query_points, fmaps=fmaps, iters=coarse_iters, down_ratio=self.coarse_down_ratio
        )
        coarse_pred_track = coarse_pred_track_lists[-1]

        if inference:
            torch.cuda.empty_cache()

        if fine_tracking:
            # Refine the coarse prediction
            fine_pred_track, pred_score = refine_track(
                images, self.fine_fnet, self.fine_predictor, coarse_pred_track, compute_score=False, chunk=fine_chunk
            )

            if inference:
                torch.cuda.empty_cache()
        else:
            fine_pred_track = coarse_pred_track
            pred_score = torch.ones_like(pred_vis)

        return fine_pred_track, coarse_pred_track, pred_vis, pred_score

    def process_images_to_fmaps(self, images):
        """
        This function processes images for inference.

        Args:
            images (torch.Tensor): The images to be processed with shape S x 3 x H x W.

        Returns:
            torch.Tensor: The processed feature maps.
        """
        if self.coarse_down_ratio > 1:
            # whether or not scale down the input images to save memory
            fmaps = self.coarse_fnet(
                F.interpolate(images, scale_factor=1 / self.coarse_down_ratio, mode="bilinear", align_corners=True)
            )
        else:
            fmaps = self.coarse_fnet(images)

        return fmaps
