# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, Linear, build_activation_layer
from mmcv.runner import BaseModule

from ...ops import resize
from ..builder import HEADS
from .decode_head import DepthBaseDecodeHead


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return x


class HeadDepth(nn.Module):
    def __init__(self, features):
        super(HeadDepth, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.head(x)
        return x


class ReassembleBlocks(BaseModule):
    """ViTPostProcessBlock, process cls_token in ViT backbone output and
    rearrange the feature vector to feature map.
    Args:
        in_channels (int): ViT feature channels. Default: 768.
        out_channels (List): output channels of each stage.
            Default: [96, 192, 384, 768].
        readout_type (str): Type of readout operation. Default: 'ignore'.
        patch_size (int): The patch size. Default: 16.
        init_cfg (dict, optional): Initialization config dict. Default: None.
    """

    def __init__(
        self, in_channels=768, out_channels=[96, 192, 384, 768], readout_type="ignore", patch_size=16, init_cfg=None
    ):
        super(ReassembleBlocks, self).__init__(init_cfg)

        assert readout_type in ["ignore", "add", "project"]
        self.readout_type = readout_type
        self.patch_size = patch_size

        self.projects = nn.ModuleList(
            [
                ConvModule(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=1,
                    act_cfg=None,
                )
                for out_channel in out_channels
            ]
        )

        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1
                ),
            ]
        )
        if self.readout_type == "project":
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(Linear(2 * in_channels, in_channels), build_activation_layer(dict(type="GELU")))
                )

    def forward(self, inputs):
        assert isinstance(inputs, list)
        out = []
        for i, x in enumerate(inputs):
            assert len(x) == 2
            x, cls_token = x[0], x[1]
            feature_shape = x.shape
            if self.readout_type == "project":
                x = x.flatten(2).permute((0, 2, 1))
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
                x = x.permute(0, 2, 1).reshape(feature_shape)
            elif self.readout_type == "add":
                x = x.flatten(2) + cls_token.unsqueeze(-1)
                x = x.reshape(feature_shape)
            else:
                pass
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
        return out


class PreActResidualConvUnit(BaseModule):
    """ResidualConvUnit, pre-activate residual unit.
    Args:
        in_channels (int): number of channels in the input feature map.
        act_cfg (dict): dictionary to construct and config activation layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        stride (int): stride of the first block. Default: 1
        dilation (int): dilation rate for convs layers. Default: 1.
        init_cfg (dict, optional): Initialization config dict. Default: None.
    """

    def __init__(self, in_channels, act_cfg, norm_cfg, stride=1, dilation=1, init_cfg=None):
        super(PreActResidualConvUnit, self).__init__(init_cfg)

        self.conv1 = ConvModule(
            in_channels,
            in_channels,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            bias=False,
            order=("act", "conv", "norm"),
        )

        self.conv2 = ConvModule(
            in_channels,
            in_channels,
            3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            bias=False,
            order=("act", "conv", "norm"),
        )

    def forward(self, inputs):
        inputs_ = inputs.clone()
        x = self.conv1(inputs)
        x = self.conv2(x)
        return x + inputs_


class FeatureFusionBlock(BaseModule):
    """FeatureFusionBlock, merge feature map from different stages.
    Args:
        in_channels (int): Input channels.
        act_cfg (dict): The activation config for ResidualConvUnit.
        norm_cfg (dict): Config dict for normalization layer.
        expand (bool): Whether expand the channels in post process block.
            Default: False.
        align_corners (bool): align_corner setting for bilinear upsample.
            Default: True.
        init_cfg (dict, optional): Initialization config dict. Default: None.
    """

    def __init__(self, in_channels, act_cfg, norm_cfg, expand=False, align_corners=True, init_cfg=None):
        super(FeatureFusionBlock, self).__init__(init_cfg)

        self.in_channels = in_channels
        self.expand = expand
        self.align_corners = align_corners

        self.out_channels = in_channels
        if self.expand:
            self.out_channels = in_channels // 2

        self.project = ConvModule(self.in_channels, self.out_channels, kernel_size=1, act_cfg=None, bias=True)

        self.res_conv_unit1 = PreActResidualConvUnit(in_channels=self.in_channels, act_cfg=act_cfg, norm_cfg=norm_cfg)
        self.res_conv_unit2 = PreActResidualConvUnit(in_channels=self.in_channels, act_cfg=act_cfg, norm_cfg=norm_cfg)

    def forward(self, *inputs):
        x = inputs[0]
        if len(inputs) == 2:
            if x.shape != inputs[1].shape:
                res = resize(inputs[1], size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
            else:
                res = inputs[1]
            x = x + self.res_conv_unit1(res)
        x = self.res_conv_unit2(x)
        x = resize(x, scale_factor=2, mode="bilinear", align_corners=self.align_corners)
        x = self.project(x)
        return x


@HEADS.register_module()
class DPTHead(DepthBaseDecodeHead):
    """Vision Transformers for Dense Prediction.
    This head is implemented of `DPT <https://arxiv.org/abs/2103.13413>`_.
    Args:
        embed_dims (int): The embed dimension of the ViT backbone.
            Default: 768.
        post_process_channels (List): Out channels of post process conv
            layers. Default: [96, 192, 384, 768].
        readout_type (str): Type of readout operation. Default: 'ignore'.
        patch_size (int): The patch size. Default: 16.
        expand_channels (bool): Whether expand the channels in post process
            block. Default: False.
    """

    def __init__(
        self,
        embed_dims=768,
        post_process_channels=[96, 192, 384, 768],
        readout_type="ignore",
        patch_size=16,
        expand_channels=False,
        **kwargs
    ):
        super(DPTHead, self).__init__(**kwargs)

        self.in_channels = self.in_channels
        self.expand_channels = expand_channels
        self.reassemble_blocks = ReassembleBlocks(embed_dims, post_process_channels, readout_type, patch_size)

        self.post_process_channels = [
            channel * math.pow(2, i) if expand_channels else channel for i, channel in enumerate(post_process_channels)
        ]
        self.convs = nn.ModuleList()
        for channel in self.post_process_channels:
            self.convs.append(ConvModule(channel, self.channels, kernel_size=3, padding=1, act_cfg=None, bias=False))
        self.fusion_blocks = nn.ModuleList()
        for _ in range(len(self.convs)):
            self.fusion_blocks.append(FeatureFusionBlock(self.channels, self.act_cfg, self.norm_cfg))
        self.fusion_blocks[0].res_conv_unit1 = None
        self.project = ConvModule(self.channels, self.channels, kernel_size=3, padding=1, norm_cfg=self.norm_cfg)
        self.num_fusion_blocks = len(self.fusion_blocks)
        self.num_reassemble_blocks = len(self.reassemble_blocks.resize_layers)
        self.num_post_process_channels = len(self.post_process_channels)
        assert self.num_fusion_blocks == self.num_reassemble_blocks
        assert self.num_reassemble_blocks == self.num_post_process_channels
        self.conv_depth = HeadDepth(self.channels)

    def forward(self, inputs, img_metas):
        assert len(inputs) == self.num_reassemble_blocks
        x = [inp for inp in inputs]
        x = self.reassemble_blocks(x)
        x = [self.convs[i](feature) for i, feature in enumerate(x)]
        out = self.fusion_blocks[0](x[-1])
        for i in range(1, len(self.fusion_blocks)):
            out = self.fusion_blocks[i](out, x[-(i + 1)])
        out = self.project(out)
        out = self.depth_pred(out)
        return out
