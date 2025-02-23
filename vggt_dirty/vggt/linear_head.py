# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# linear head implementation for DUST3R
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from .head_act import activate_head

class LinearHead (nn.Module):
    """
    """
    def __init__(self,
                 dim_in,
                 patch_size=16,
                 output_dim = 4,
                 normalize_act="norm_exp",
                 intermediate_layer_idx = None,
                 **kwargs,
                 ):
        super().__init__()
        self.patch_size = patch_size
        self.output_dim = output_dim
        self.intermediate_layer_idx = intermediate_layer_idx

        self.norm = nn.LayerNorm(dim_in)
        if intermediate_layer_idx is not None:
            self.proj = nn.Linear(dim_in * len(intermediate_layer_idx), (output_dim)*self.patch_size**2)
        else:
            self.proj = nn.Linear(dim_in, (output_dim)*self.patch_size**2)

        self.normalize_act = normalize_act

    def forward(self, aggregated_tokens_list, batch, patch_start_idx):
        B, _, _, H, W = batch["images"].shape
        S = aggregated_tokens_list[0].shape[1]


        if self.intermediate_layer_idx is not None:
            tokens = []
            for layer_idx in self.intermediate_layer_idx:   
                tokens.append(self.norm(aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]))
            
            tokens = torch.cat(tokens, dim=-1)
            _, _, P, C = tokens.shape
            tokens = tokens.view(B*S, P, C)
            feat = self.proj(tokens)
        else:
            tokens = aggregated_tokens_list[-1]
            tokens = tokens[:, :, patch_start_idx:]
            _, _, P, C = tokens.shape
            tokens = tokens.view(B*S, P, C)
            feat = self.proj(self.norm(tokens))

        feat = feat.transpose(-1, -2).view(B * S, -1, H//self.patch_size, W//self.patch_size)
        feat = F.pixel_shuffle(feat, self.patch_size)  # B,3,H,W


        pts3d, conf = activate_head(feat, normalize_act=self.normalize_act)

        # back to B, S
        # B, S, H, W, 3
        pts3d = pts3d.view(B, S, *pts3d.shape[1:])
        # B, S, H, W
        conf = conf.view(B, S, *conf.shape[1:])

        return pts3d, conf
