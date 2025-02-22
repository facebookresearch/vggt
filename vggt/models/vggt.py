import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Any, Dict, List, Optional, Tuple, Union

# from off3d.models.vggt.utils import random_mask_single_patch_vectorized  # Removed unused import
from hydra.utils import instantiate
# from .loss import *

def configure_dict(module, **attributes):
    if module:
        for attr, value in attributes.items():
            setattr(module, attr, value)


class VGGT(nn.Module):
    def __init__(self,
                AGGREGATOR: Dict,
                CameraHead: Dict,
                PointHead: Dict,
                DepthHead: Dict,
                MatchHead: Dict,
                TrackHead: Dict,
                num_register_tokens,
                init_values,
                qk_norm,
                ffn_layer,
                patch_size,
                enable_head_mp=False,
                **kwargs):
        super().__init__()

        config_attrs = {
            'patch_size': patch_size,
            'init_values': init_values,
            'qk_norm': qk_norm,
            'ffn_layer': ffn_layer,
            'num_register_tokens': num_register_tokens
        }


        if AGGREGATOR:
            configure_dict(AGGREGATOR, **config_attrs)
            self.aggregator = instantiate(AGGREGATOR, _recursive_=False)
        else:
            self.aggregator = None

        if CameraHead:
            configure_dict(CameraHead, **config_attrs)
            CameraHead.loss_kwargs.pose_encoding_type = CameraHead.pose_encoding_type
            self.camera_head_loss_kwargs = CameraHead.loss_kwargs
            self.camera_head = instantiate(CameraHead, _recursive_=False)
        else:
            self.camera_head = None

        if PointHead:
            configure_dict(PointHead, **config_attrs)
            self.point_head_loss_kwargs = PointHead.loss_kwargs
            self.point_head = instantiate(PointHead, _recursive_=False)
        else:
            self.point_head = None

        if DepthHead:
            configure_dict(DepthHead, **config_attrs)
            self.depth_head_loss_kwargs = DepthHead.loss_kwargs
            self.depth_head = instantiate(DepthHead, _recursive_=False)
        else:
            self.depth_head = None

        if MatchHead:
            configure_dict(MatchHead, **config_attrs)
            self.match_head_loss_kwargs = MatchHead.loss_kwargs
            self.match_head = instantiate(MatchHead, _recursive_=False)
        else:
            self.match_head = None

        if TrackHead:
            configure_dict(TrackHead, **config_attrs)
            self.track_head_loss_kwargs = TrackHead.loss_kwargs
            self.track_head = instantiate(TrackHead, _recursive_=False)
        else:
            self.track_head = None

        self.enable_head_mp = enable_head_mp
        # self.mask_patch_ratio = mask_patch_ratio
        # self.mask_patch_size = mask_patch_size
        
        
    def forward(self, batch, device=None):
        images = (batch["images"]) #.to(device) # B x S x 3 x H x W
        # intrinsics = (batch["intrinsics"])#.to(device)
        # extrinsics = (batch["extrinsics"])#.to(device)
        B, S, C, H, W = images.shape


        # if self.training and self.mask_patch_ratio > 0:  # Commented out masking
        #     for _ in range(1000):
        #         print("Please do not use mask_patch_ratio for now")

        # predictions = {}  # Removed redundant dict

        aggregated_tokens_list, _, patch_start_idx = self.aggregator(images, batch=batch)


        # Pose branch
        # TODO check pose encoding conversion  # Removed TODO
        # loss = 0
        

        predictions = {}
        


        # well by default we use amp for track head
        if self.track_head is not None:
            track_loss_dict = self.track_head(aggregated_tokens_list, batch=batch, patch_start_idx=patch_start_idx)
            predictions.update(track_loss_dict)


        with torch.cuda.amp.autocast(enabled=self.enable_head_mp):
            if self.camera_head is not None:
                pred_pose_enc_list = self.camera_head(aggregated_tokens_list, batch=batch, patch_start_idx=patch_start_idx)
                camera_loss_dict = {}
                camera_loss_dict["pred_extrinsic_list"] = pred_pose_enc_list
                # with torch.cuda.amp.autocast(enabled=False):
                #     if not isinstance(pred_pose_enc_list, dict):
                #         camera_loss_dict, last_pred_extrinsic = camera_loss(pred_pose_enc_list, batch, **self.camera_head_loss_kwargs)
                #         predictions["pred_extrinsic"] = last_pred_extrinsic
                #     else:
                #         camera_loss_dict = pred_pose_enc_list
                predictions.update(camera_loss_dict)
            
            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(aggregated_tokens_list, batch=batch, patch_start_idx=patch_start_idx)
                # with torch.cuda.amp.autocast(enabled=False):
                #     pts3d_loss_dict = point_loss(pts3d, pts3d_conf, batch, **self.point_head_loss_kwargs)
                # predictions.update(pts3d_loss_dict)
                predictions["pred_world_points"] = pts3d
                predictions["pred_world_points_conf"] = pts3d_conf

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(aggregated_tokens_list, batch=batch, patch_start_idx=patch_start_idx)
                # with torch.cuda.amp.autocast(enabled=False):
                #     depth_loss_dict = depth_loss(depth, depth_conf, batch, **self.depth_head_loss_kwargs)
                # predictions.update(depth_loss_dict)
                predictions["pred_depth"] = depth
                predictions["pred_depth_conf"] = depth_conf

            if self.match_head is not None:
                match_loss_dict = self.match_head(aggregated_tokens_list, batch=batch, patch_start_idx=patch_start_idx)
                predictions.update(match_loss_dict)
                
        predictions.update(batch)

        return predictions
