# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate

from off3d.models.dino_layers.block import Block
from off3d.models.dino_layers import Mlp
from off3d.models.vggt.utils import PoseEmbedding
from off3d.models.vggt.head_act import activate_pose

def modulate(x, shift, scale):
    # modified from https://github.com/facebookresearch/DiT/blob/796c29e532f47bba17c5b9c5eb39b9354b8b7c64/models.py#L19
    return x * (1 + scale) + shift



class CameraHead(nn.Module):
    def __init__(
        self,
        dim_in=2048,
        patch_size=14,
        qk_norm=False,
        trunk_depth=4,
        new_trunk=True,
        update_new_trunk_tokens=False,
        pose_encoding_type="absT_quaR_FoV",
        proj_dim=-1,
        num_heads=16,
        mlp_ratio=4,
        init_values=None,
        act_dict=None,
        **kwargs,
    ):
        super().__init__()
        
        #  Three types:
        # 1. Linear projection
        # 2. New trunk
        # 3. Old trunk
        
        self.new_trunk = new_trunk
        if pose_encoding_type=="absT_quaR_FoV":
            self.target_dim = 9
        elif pose_encoding_type=="absT_quaR_OneFLM1":
            self.target_dim = 8
        else:
            raise ValueError(f"Unsupported pose encoding type: {pose_encoding_type}")
        
        self.update_new_trunk_tokens = update_new_trunk_tokens
        self.act_dict = act_dict
        self.trunk_depth = trunk_depth

        self.token_norm = nn.LayerNorm(dim_in)

        if proj_dim > 0:
            self.proj = nn.Linear(dim_in, proj_dim)
            dim_in = proj_dim
        else:
            self.proj = nn.Identity()
        
        if self.trunk_depth <0:
            self.pose_branch = nn.Linear(dim_in, self.target_dim)
        else:
            self.trunk = nn.Sequential(
                *[
                    Block(
                        dim=dim_in,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qk_norm=qk_norm,
                        init_values=init_values,
                    )
                    for _ in range(trunk_depth)
                ]
            )
            self.trunk_norm = nn.LayerNorm(dim_in)

            if self.new_trunk:
                # TODO: self.empty_pose_tokens -> BxSxC
                self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 1, self.target_dim))
                self.embed_pose = nn.Linear(self.target_dim, dim_in)
                
                self.poseLN_modulation = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(dim_in, 3 * dim_in, bias=True)
                )
                
                self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)
                self.pose_branch = Mlp(
                    in_features=dim_in,
                    hidden_features=dim_in // 2,
                    out_features=self.target_dim,
                    drop=0,
                )
            else:
                self.ffeat_norm = nn.LayerNorm(dim_in)
                self.pose_branch = Mlp(
                    in_features=dim_in,
                    hidden_features=dim_in * 2,
                    out_features=dim_in + self.target_dim,
                    drop=0,
                )

                self.ffeat_updater = nn.Sequential(
                    nn.Linear(dim_in, dim_in), nn.GELU()
                )

                # sine and cosine embed for camera parameters
                self.embed_pose = PoseEmbedding(
                    target_dim=self.target_dim,
                    n_harmonic_functions=(dim_in // self.target_dim) // 2,
                    append_input=False,
                )
                self.embed_pose_proj = nn.Linear(self.embed_pose.out_dim, dim_in)


    def forward(self, aggregated_tokens_list, batch, patch_start_idx, iters=4,):
        """
        """
        tokens = aggregated_tokens_list[-1]
        # only use the Pose token for camera prediction
        pose_tokens = tokens[:, :, 0]
        pose_tokens = self.token_norm(pose_tokens)
        pose_tokens = self.proj(pose_tokens)
        
        B, S, C = pose_tokens.shape
        
        if self.trunk_depth < 0:
            pred_pose_enc = self.pose_branch(pose_tokens)
            pred_pose_enc_list = [activate_pose(pred_pose_enc, **self.act_dict)]
        elif self.new_trunk:
            pred_pose_enc_list = self.new_trunk_fn(pose_tokens, iters)
        else:
            pred_pose_enc_list = self.old_trunk_fn(pose_tokens, iters)


        # TODO add act here
        return pred_pose_enc_list


    def new_trunk_fn(self, pose_tokens, iters):
        B, S, C = pose_tokens.shape
        
        pred_pose_enc = None
        pose_tokens_init = pose_tokens.clone()
        
        pred_pose_enc_list = []

        for iter_num in range(iters):
            if pred_pose_enc is None:
                # model_input = self.empty_representation BxSxC
                module_input = self.embed_pose(self.empty_pose_tokens.expand(B, S, -1))
            else:
                pred_pose_enc = pred_pose_enc.detach()
                module_input = self.embed_pose(pred_pose_enc)

            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)
            pose_tokens_modulated = gate_msa * modulate(self.adaln_norm(pose_tokens), shift_msa, scale_msa)
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens

            pose_tokens_modulated = self.trunk(pose_tokens_modulated)
            pred_pose_enc_delta = self.pose_branch(self.trunk_norm(pose_tokens_modulated))

            if pred_pose_enc is None:
                pred_pose_enc = pred_pose_enc_delta
            else:
                pred_pose_enc = pred_pose_enc + pred_pose_enc_delta
            
            if self.update_new_trunk_tokens:
                pose_tokens = pose_tokens_modulated + pose_tokens_init
            
            pred_pose_enc_list.append(activate_pose(pred_pose_enc, **self.act_dict))
            
        return pred_pose_enc_list


    def old_trunk_fn(self, pose_tokens, iters):
        B, S, C = pose_tokens.shape

        pred_pose_enc = torch.zeros(B, S, self.target_dim).to(
            pose_tokens.device
        )

        pose_tokens_init = pose_tokens.clone()
        
        pred_pose_enc_list = []
        
        for iter_num in range(iters):            
            pred_pose_enc = pred_pose_enc.detach()

            # Embed the camera parameters and add to pose_tokens
            pose_embed = self.embed_pose_proj(self.embed_pose(pred_pose_enc))
            pose_tokens = pose_tokens + pose_embed

            # Run trunk transformers on pose_tokens
            pose_tokens = self.trunk(pose_tokens)

            # Predict the delta feat and pose encoding at each iteration
            delta = self.pose_branch(self.trunk_norm(pose_tokens))
            delta_pred_pose_enc = delta[..., : self.target_dim]
            delta_feat = delta[..., self.target_dim :]

            pose_tokens = self.ffeat_updater(self.ffeat_norm(delta_feat)) + pose_tokens

            pred_pose_enc = pred_pose_enc + delta_pred_pose_enc
            pose_tokens = (pose_tokens + pose_tokens_init) / 2
            pred_pose_enc_list.append(activate_pose(pred_pose_enc, **self.act_dict))
            
        return pred_pose_enc_list
    
