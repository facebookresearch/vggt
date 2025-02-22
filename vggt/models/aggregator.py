# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging

import pdb
import math
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.utils import instantiate
from torch.nn.init import trunc_normal_

from torch.utils.checkpoint import checkpoint
from omegaconf import OmegaConf
from contextlib import nullcontext

from typing import Any, Dict, List, Optional, Tuple, Union

# from off3d.utils.train_utils import remove_if_not_match

# from off3d.models.modules import AttnBlock, CrossAttnBlock, Mlp, ResidualBlock, RoPEAttnBlock
# from vggsfm.models.utils import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid
# from off3d.models.dino_layers import SwiGLUFFNFused, PatchEmbed

from vggt.layers import SwiGLUFFNFused, PatchEmbed
from vggt.layers.block import Block

# from off3d.models.dino_layers.block import Block
# from vggt.layers.rope import RoPE2D, PositionGetter
from vggt.layers.rope import RoPE2D, PositionGetter

# from off3d.models.multihead_with_qk_norm import MultiheadAttention_with_qk_norm
# from off3d.models.rope import RoPEMulitheadAttention

from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2


logger = logging.getLogger(__name__)



_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]




class Aggregator(nn.Module):
    def __init__(
        self,
        image_size = 512,
        patch_size = 16,
        num_register_tokens = 4,
        image_backbone = "dinov2_vitl14_reg",
        aa_block_size = 1,
        aa_layer_size = 24,
        aa_block_kwargs = Dict, 
        attn_block = Block,
        aa_order = ["frame", "global"],
        use_checkpoint = False,
        use_reentrant = False, 
        use_dino_tokens = False,
        use_patch_tokens_only = False,
        freeze_dino=False,
        freeze_dino_inter=False,
        # pose_embed=False,
        embed_type="no",
        patch_embed_by_conv=False,
        decoder_load_dino=False,
        backbone_qk_norm=False,
        **kwargs,
    ):
        super().__init__()

        if image_backbone is None:
            self.image_backbone = None
        else:
            self.__build_image_backbone__(image_backbone, image_size, 
                                          patch_size, num_register_tokens, freeze_dino=freeze_dino,
                                          freeze_dino_inter=freeze_dino_inter, backbone_qk_norm=backbone_qk_norm)
        
    
        self.freeze_dino = freeze_dino
        
        if use_checkpoint and not freeze_dino:
            self.image_backbone.use_checkpoint = True
        else:
            self.image_backbone.use_checkpoint = False
            
        self.image_backbone.use_reentrant = use_reentrant
        
        if aa_block_kwargs['rope_freq']>0:
            self.rope = RoPE2D(freq=aa_block_kwargs['rope_freq'])
            self.position_getter = PositionGetter()
        else:
            self.rope = None
        
        frame_blocks_list = []
        global_blocks_list = []
        for _ in range(aa_layer_size):
            frame_blocks_list.append(attn_block(**aa_block_kwargs, rope=self.rope))
            global_blocks_list.append(attn_block(**aa_block_kwargs, rope=self.rope))
        
        self.frame_blocks = nn.ModuleList(frame_blocks_list)
        self.global_blocks = nn.ModuleList(global_blocks_list)
        
        if "mlp" in embed_type:
            self.register_mlp = nn.ModuleList([nn.Linear(aa_block_kwargs['dim'], aa_block_kwargs['dim']) for _ in range(aa_layer_size)])
        
        self.aa_order = aa_order
        self.aa_block_size = aa_block_size
        self.aa_layer_size = aa_layer_size
        
        assert self.aa_layer_size % self.aa_block_size == 0, "aa_layer_size must be divisible by aa_block_size"
        self.aa_block_num = self.aa_layer_size // self.aa_block_size
        
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint
        self.use_reentrant = use_reentrant
        self.use_dino_tokens = use_dino_tokens
        self.use_patch_tokens_only = use_patch_tokens_only
        # self.pose_embed = pose_embed
        # self.register_embed = register_embed
        self.embed_type = embed_type
        
        if self.use_patch_tokens_only:
            self.query_ref_token = nn.Parameter(torch.randn(1, 2, 1, aa_block_kwargs['dim']))
            self.patch_start_idx = 0
            nn.init.normal_(self.query_ref_token, std=1e-6)
        elif self.use_dino_tokens:
            # One for query frame and one for other frames
            self.query_ref_token = nn.Parameter(torch.randn(1, 2, 1, aa_block_kwargs['dim']))
            self.patch_start_idx = 1 + num_register_tokens + 1
            nn.init.normal_(self.query_ref_token, std=1e-6)
        else:
            self.pose_token = nn.Parameter(torch.randn(1, 2, 1, aa_block_kwargs['dim']))
            self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, aa_block_kwargs['dim']))
            self.patch_start_idx = 1 + num_register_tokens
            nn.init.normal_(self.pose_token, std=1e-6)
            nn.init.normal_(self.register_token, std=1e-6)


        if decoder_load_dino:   
            dinov2_weights = self.image_backbone.state_dict()
            decoder_dinov2_weights = dino_to_aggregator(dinov2_weights)
            missing_keys, unexpected_keys = self.load_state_dict(decoder_dinov2_weights, strict=False)
            print(f"missing_keys for decoder_load_dino: {missing_keys}")
            print(f"unexpected_keys for decoder_load_dino: {unexpected_keys}")
        
        if patch_embed_by_conv:
            self.image_backbone = self.image_backbone.patch_embed


        for name, value in (
            ("_resnet_mean", _RESNET_MEAN),
            ("_resnet_std", _RESNET_STD),
        ):
            self.register_buffer(
                name,
                torch.FloatTensor(value).view(1, 1, 3, 1, 1),
                persistent=False,
            )


    def __build_image_backbone__(self, image_backbone, image_size, patch_size, num_register_tokens,
                                interpolate_antialias=True,
                                interpolate_offset=0.0,
                                block_chunks=0,
                                init_values=1.0,
                                freeze_dino=False,
                                freeze_dino_inter=False,
                                backbone_qk_norm=False,
                                ):
        
        vit_models = { "dinov2_vitl14_reg": vit_large, 
                      "dinov2_vitb14_reg": vit_base, 
                      "dinov2_vits14_reg": vit_small, 
                      "dinov2_vitg2_reg": vit_giant2,
                      } 
        
        if image_backbone not in vit_models: 
            raise NotImplementedError 
        
        self.image_backbone = vit_models[image_backbone](img_size=image_size, 
                                patch_size=patch_size, num_register_tokens=num_register_tokens, 
                                interpolate_antialias=interpolate_antialias, 
                                interpolate_offset=interpolate_offset, 
                                block_chunks=block_chunks, init_values=init_values, qk_norm=backbone_qk_norm)

        pretrained_model = torch.hub.load("facebookresearch/dinov2", image_backbone)
        pretrained_model_dict = pretrained_model.state_dict()
        image_backbone_dict = self.image_backbone.state_dict()

        all_pretrained_keys = list(pretrained_model_dict.keys())

        for cur_key in all_pretrained_keys:     
            pretrained_model_dict = remove_if_not_match(image_backbone_dict, pretrained_model_dict, cur_key)

        missing_keys, unexpected_keys = self.image_backbone.load_state_dict(pretrained_model_dict, strict=False)
        
        self.image_backbone.mask_token.requires_grad_(False)
        self.image_backbone.freeze_dino = freeze_dino
        
        if freeze_dino:
            print("Freezing DINO layers")
            for name, param in self.image_backbone.named_parameters():
                param.requires_grad_(False)

        if freeze_dino_inter:
            print("Freezing DINO intermediate layers")
            for name, param in self.image_backbone.named_parameters():
                if name not in ['pos_embed', 'patch_embed.proj.weight']:
                    param.requires_grad_(False)

            
        print("Loading pretrained DINO v2 model: ")
        print(f"missing_keys: {missing_keys}")
        print("Loading pretrained DINO v2 model: ")
        print(f"unexpected_keys: {unexpected_keys}")

        
    def forward(
        self, images, 
        masks=None,
        batch=None,
    ):
        """
        TODO List:
        
        """
        
        # The input images are in the range of [0, 1]
        B, S, C_in, H, W = images.shape
        device = images.device


        images = (images - self._resnet_mean) / self._resnet_std
        
        
        if self.image_backbone is not None:
            images = images.view(B * S, C_in, H, W)

            with torch.no_grad() if self.freeze_dino else nullcontext():
                backbone_output = self.image_backbone(images)
                
            if isinstance(backbone_output, dict):
                patch_tokens = backbone_output["x_norm_patchtokens"]
            else:
                patch_tokens = backbone_output
                
            BS, P, C = patch_tokens.shape
            
            if self.use_patch_tokens_only:
                indicator_tokens = slice_expand_and_flatten(self.query_ref_token, B, S)
                tokens = patch_tokens + indicator_tokens
            elif self.use_dino_tokens:
                dino_cls_token = backbone_output["x_norm_clstoken"][:, None] # BS, 1, C
                dino_register_tokens = backbone_output["x_norm_regtokens"] # BS, num_register_tokens, C
                
                indicator_tokens = slice_expand_and_flatten(self.query_ref_token, B, S)
                tokens = torch.cat([dino_cls_token, dino_register_tokens, indicator_tokens, patch_tokens], dim=1)
            else:
                # B, S, P, C
                pose_token = slice_expand_and_flatten(self.pose_token, B, S)
                register_token = slice_expand_and_flatten(self.register_token, B, S)
                
                tokens = torch.cat([pose_token, register_token, patch_tokens], dim=1)
        else:
            # well well I need to write this, hopefully in the near future
            raise NotImplementedError


        if self.rope is not None:
            pos = self.position_getter(B*S, H//self.patch_size, W//self.patch_size, device=device)
        else:
            pos = None
            
        

        if self.patch_start_idx > 0:
            # shift the position by 1 so that the special tokens are at 0
            pos = pos + 1
            pos_special = torch.zeros(B*S, self.patch_start_idx, 2).to(device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
        

        _, P, C = tokens.shape


        frame_idx = 0
        global_idx = 0
        output_list = []
        

        for aa_block_idx in range(self.aa_block_num):            
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, self.aa_block_size, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, self.aa_block_size, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")
            
            
            # for frame_inter, global_inter in zip(frame_intermediates, global_intermediates):
            #     concat_inter = torch.cat([frame_inter, global_inter], dim=-1)  # [B x S x P x 2C]
            #     output_list.append(concat_inter)

            for i in range(len(frame_intermediates)):
                # [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)


        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_list, None, self.patch_start_idx


   
    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, num_blocks, pos=None):
        """
        Process frame attention blocks.
        """
        if tokens.shape != (B*S, P, C):
            tokens = tokens.view(B, S, P, C)
            tokens = tokens.view(B*S, P, C)
            
        if pos is not None and pos.shape != (B*S, P, 2):
            pos = pos.view(B, S, P, 2)
            pos = pos.view(B*S, P, 2)
        
        intermediates = []
        
        for _ in range(num_blocks):
            if self.use_checkpoint:
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))
            
        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, num_blocks, pos=None):
        """
        Process global attention blocks.
        """
        # pose_embed
        
        if tokens.shape != (B, S*P, C):
            tokens = tokens.view(B, S, P, C)
                    
        
        ############################################################
        # Frame embedding
        if "register" in self.embed_type:
            embed_tokens = tokens[:, :, 1:2, ...].clone() 
        if "gauss" in self.embed_type:
            embed_tokens = torch.randn((B, S, 1, C),device=tokens.device, dtype=tokens.dtype)
            
        if self.embed_type != "no":   
            embed_tokens = F.normalize(embed_tokens, dim=-1)
            
        if "mlp" in self.embed_type:
            embed_tokens = self.register_mlp[global_idx](embed_tokens)
            
        if "mlpnorm" in self.embed_type:
            embed_tokens = F.normalize(embed_tokens, dim=-1)
        if "all" in self.embed_type:
            tokens = tokens + embed_tokens
        elif "part" in self.embed_type:
            tokens[:, :, self.patch_start_idx:] = tokens[:, :, self.patch_start_idx:] + embed_tokens
        else:
            assert self.embed_type == "no"
            
        if "postnorm" in self.embed_type:
            tokens = F.normalize(tokens, dim=-1)
            # tokens = self.embed_norm(tokens)
        ############################################################


        
        tokens = tokens.view(B, S*P, C)
            
        if pos is not None and pos.shape != (B, S*P, 2):
            pos = pos.view(B, S, P, 2)
            pos = pos.view(B, S*P, 2)
        
        intermediates = []
        for _ in range(num_blocks):
            if self.use_checkpoint:
                tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))
            
        return tokens, global_idx, intermediates




def slice_expand_and_flatten(token_tensor, B, S):
    """
    1) Takes the first token (index=0) and the remaining tokens (index=1..S-1).
    2) Expands them along batch dimension B.
    3) Concatenates along the time/sequence dimension => (B, S, ...).
    4) Flattens the first two dims to produce => (B*S, ...).

    Args:
        token_tensor: a tensor expected to have shape (1, S, ...) or (some_batch, S, ...).
                      We'll slice along dim=1.
        B: batch size.
        S: number of frames/time-steps.

    Returns:
        Flattened token tensor of shape (B*S, ...).
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined




def dino_to_aggregator(dinov2_weights):
    new_dinov2_weights = {}
    for key, value in dinov2_weights.items():
        if "blocks" in key:
            for new_attn_key in ["frame_blocks", "global_blocks"]:
                new_key = key.replace("blocks", new_attn_key)
                # if 'attn' in key:
                    # if "qkv.weight" in key:
                    #     new_key = new_key.replace('qkv.weight', 'in_proj_weight')
                    # elif "qkv.bias" in key:
                    #     new_key = new_key.replace('qkv.bias', 'in_proj_bias')
                    # elif 'proj.weight' in key:
                    #     new_key = new_key.replace('proj.weight', 'out_proj.weight')
                    # elif 'proj.bias' in key:
                    #     new_key = new_key.replace('proj.bias', 'out_proj.bias')
                new_dinov2_weights[new_key] = value.clone()
    return new_dinov2_weights

    


def remove_if_not_match(model_state_dict, state_dict, key):
    if key in state_dict.keys() and key in model_state_dict.keys():
        if state_dict[key].shape != model_state_dict[key].shape:
            print(f"Warning: {key} shape mismatch, removing it")
            del state_dict[key]
    return state_dict
