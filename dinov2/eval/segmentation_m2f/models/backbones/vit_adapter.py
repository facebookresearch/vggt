# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import BACKBONES
from torch.nn.init import normal_

from ...ops.modules import MSDeformAttn
from .adapter_modules import InteractionBlock, InteractionBlockWithCls, SpatialPriorModule, deform_inputs
from .vit import TIMMVisionTransformer


@BACKBONES.register_module()
class ViTAdapter(TIMMVisionTransformer):
    def __init__(
        self,
        pretrain_size=224,
        num_heads=12,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=6,
        init_values=0.0,
        interaction_indexes=None,
        with_cffn=True,
        cffn_ratio=0.25,
        deform_ratio=1.0,
        add_vit_feature=True,
        pretrained=None,
        use_extra_extractor=True,
        freeze_vit=False,
        use_cls=True,
        with_cp=False,
        *args,
        **kwargs
    ):

        super().__init__(num_heads=num_heads, pretrained=pretrained, with_cp=with_cp, *args, **kwargs)
        if freeze_vit:
            for param in self.parameters():
                param.requires_grad = False

        # self.num_classes = 80
        self.use_cls = use_cls
        if not self.use_cls:
            self.cls_token = None
        self.num_block = len(self.blocks)
        self.pretrain_size = (pretrain_size, pretrain_size)
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        embed_dim = self.embed_dim

        block_fn = InteractionBlockWithCls if use_cls else InteractionBlock

        self.level_embed = nn.Parameter(torch.zeros(3, embed_dim))
        self.spm = SpatialPriorModule(inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False)
        self.interactions = nn.Sequential(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=deform_num_heads,
                    n_points=n_points,
                    init_values=init_values,
                    drop_path=self.drop_path_rate,
                    norm_layer=self.norm_layer,
                    with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio,
                    deform_ratio=deform_ratio,
                    extra_extractor=((True if i == len(interaction_indexes) - 1 else False) and use_extra_extractor),
                    with_cp=with_cp,
                )
                for i in range(len(interaction_indexes))
            ]
        )
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _get_pos_embed(self, pos_embed, H, W):
        pos_embed = pos_embed.reshape(
            1, self.pretrain_size[0] // self.patch_size, self.pretrain_size[1] // self.patch_size, -1
        ).permute(0, 3, 1, 2)
        pos_embed = (
            F.interpolate(pos_embed, size=(H, W), mode="bicubic", align_corners=False)
            .reshape(1, -1, H * W)
            .permute(0, 2, 1)
        )
        return pos_embed

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x, self.patch_size)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        H_c, W_c = x.shape[2] // 16, x.shape[3] // 16
        x, H_toks, W_toks = self.patch_embed(x)
        # print("H_toks, W_toks =", H_toks, W_toks)
        bs, n, dim = x.shape
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H_toks, W_toks)
        if self.use_cls:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_token, x), dim=1)
            pos_embed = torch.cat((self.pos_embed[:, :1], pos_embed), dim=1)
        x = self.pos_drop(x + pos_embed)
        # For CLIP
        x = self.norm_pre(x)

        # Interaction
        if self.use_cls:
            cls, x = (
                x[
                    :,
                    :1,
                ],
                x[
                    :,
                    1:,
                ],
            )
        outs = list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            if self.use_cls:
                x, c, cls = layer(
                    x,
                    c,
                    cls,
                    self.blocks[indexes[0] : indexes[-1] + 1],
                    deform_inputs1,
                    deform_inputs2,
                    H_c,
                    W_c,
                    H_toks,
                    W_toks,
                )
            else:
                x, c = layer(
                    x,
                    c,
                    self.blocks[indexes[0] : indexes[-1] + 1],
                    deform_inputs1,
                    deform_inputs2,
                    H_c,
                    W_c,
                    H_toks,
                    W_toks,
                )
            outs.append(x.transpose(1, 2).view(bs, dim, H_toks, W_toks).contiguous())

        # Split & Reshape
        c2 = c[:, 0 : c2.size(1), :]
        c3 = c[:, c2.size(1) : c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1) :, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H_c * 2, W_c * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H_c, W_c).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H_c // 2, W_c // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x1, x2, x3, x4 = outs

            x1 = F.interpolate(x1, size=(4 * H_c, 4 * W_c), mode="bilinear", align_corners=False)
            x2 = F.interpolate(x2, size=(2 * H_c, 2 * W_c), mode="bilinear", align_corners=False)
            x3 = F.interpolate(x3, size=(1 * H_c, 1 * W_c), mode="bilinear", align_corners=False)
            x4 = F.interpolate(x4, size=(H_c // 2, W_c // 2), mode="bilinear", align_corners=False)
            # print(c1.shape, c2.shape, c3.shape, c4.shape, x1.shape, x2.shape, x3.shape, x4.shape, H_c, H_toks)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]
