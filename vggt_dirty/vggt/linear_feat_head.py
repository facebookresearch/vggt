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
from off3d.models.dino_layers import Mlp

class LinearFeatHead(nn.Module):
    """
    """
    def __init__(self,
                 dim_in,
                 patch_size=16,
                 output_dim = 4,
                 normalize_act="norm",
                 normalize_act_conf="expp0",
                 temperature=0.5,
                 eps = 1e-8,
                 alpha = 10,
                 **kwargs,
                 ):
        super().__init__()
        self.patch_size = patch_size
        self.output_dim = output_dim

        self.norm = nn.LayerNorm(dim_in)
        self.proj = Mlp(
                    in_features=dim_in,
                    hidden_features=dim_in * 4,
                    out_features=(output_dim)*self.patch_size**2,
                    drop=0,
                )
        self.normalize_act = normalize_act
        self.normalize_act_conf = normalize_act_conf
        
        
        self.temperature = temperature
        self.alpha = alpha
        self.eps = eps

    def forward(self, aggregated_tokens_list, batch, patch_start_idx):
        B, S, _, H, W = batch["images"].shape

        
        tokens = aggregated_tokens_list[-1]
        # ignore cls tokens, registers, and so on
        # B, S, P, C
        tokens = tokens[:, :, patch_start_idx:]

        _, _, P, C = tokens.shape

        tokens = tokens.view(B*S, P, C)

        feat = self.proj(self.norm(tokens))

        feat = feat.transpose(-1, -2).view(B * S, -1, H//self.patch_size, W//self.patch_size)
        feat = F.pixel_shuffle(feat, self.patch_size)  # B,3,H,W

        descriptor, conf = activate_head(feat, normalize_act=self.normalize_act, normalize_act_conf=self.normalize_act_conf)

        # B, S, H, W, 3
        descriptor = descriptor.view(B, S, *descriptor.shape[1:])
        # B, S, H, W
        conf = conf.view(B, S, *conf.shape[1:])

        ########################################################################################
        tracks = batch["tracks"]
        tracks = tracks.round(decimals=0).long()
        track_vis_mask = batch["track_vis_mask"]
        # track_positive_mask = batch["track_positive_mask"]
        
        uu, vv = tracks.unbind(-1) # B, S, N

        # descriptor : B, S, H, W, D
        # uu, vv : B, S, N
        
        # Sample descriptor at the track locations
        # pesudo code: descriptor[batchid, sid, vv, uu]

        b_idx = torch.arange(B, device=descriptor.device)[:, None, None]  # shape: (B, 1, 1)
        s_idx = torch.arange(S, device=descriptor.device)[None, :, None]  # shape: (1, S, 1)

        vv = vv.clamp(0, H-1)
        uu = uu.clamp(0, W-1)

        # Sample the descriptor at the track locations: (B, S, N, D)
        sampled_desc = descriptor[b_idx, s_idx, vv, uu].clone()
        sampled_conf = conf[b_idx, s_idx, vv, uu].clone()

        ########################################################################################        
        query_desc = sampled_desc[:, 0:1].expand(-1, S-1, -1, -1)
        ref_desc = sampled_desc[:, 1:]
        valid_mask = track_vis_mask[:, 1:]
        
        query_conf = sampled_conf[:, 0:1].expand(-1, S-1, -1)
        ref_conf = sampled_conf[:, 1:]
        
        query_desc = query_desc[valid_mask]
        ref_desc = ref_desc[valid_mask]
        query_conf = query_conf[valid_mask]
        ref_conf = ref_conf[valid_mask]
        
        # InfoNCE loss
        sim = get_similarities(query_desc, ref_desc, euc=False) / self.temperature
        sim[sim.isnan()] = -torch.inf  
        sim = sim.exp_()  
        positives = sim.diagonal(dim1=-2, dim2=-1)
        
        loss_info_nce = -(torch.log((positives / sim.sum(dim=-2)).clip(self.eps)) + torch.log((positives / sim.sum(dim=-1)).clip(self.eps)))
        avg_conf = (ref_conf + query_conf)/2
        conf_loss = loss_info_nce * avg_conf - self.alpha * avg_conf.log()

        if conf_loss.numel() == 0:
            conf_loss = sampled_desc * 0
            
        
        loss_dict = {
            "loss_info_nce": loss_info_nce.mean() if loss_info_nce.numel() > 0 else 0,
            "loss_conf_match": conf_loss.mean() if conf_loss.numel() > 0 else 0,
            "descriptor": descriptor,
        }

        return loss_dict




        # from off3d.data.track_util import visualize_tracks_on_images
        # visualize_tracks_on_images(
        #     images=batch["images"], 
        #     tracks=batch["tracks"], 
        #     track_vis_mask=batch["track_vis_mask"],
        #     out_dir="track_visuals"
        # )
        



    

def get_similarities(desc1, desc2, euc=False):
    if euc:  # euclidean distance in same range than similarities
        dists = (desc1[:, :, None] - desc2[:, None]).norm(dim=-1)
        sim = 1 / (1 + dists)
    else:
        # Compute similarities
        sim = desc1 @ desc2.transpose(-2, -1)
    return sim

    