# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# linear head implementation for DUST3R
# --------------------------------------------------------

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from .head_act import activate_head
from .utils import normalized_view_plane_uv, HarmonicEmbedding, position_grid_to_embed
from .dpt_head import DPTHead
from .match_head import MatchHead
from ..track_modules.base_track_predictor import BaseTrackerPredictor
from ..track_modules.base_track_predictor_v2 import BaseTrackerPredictorV2

EPS = 1e-6

def reduce_masked_mean(x, mask, dim=None, keepdim=False):
    # x and mask are the same shape, or at least broadcastably so
    # returns shape-1
    # axis can be a list of axes
    for a, b in zip(x.size(), mask.size()):
        assert a == b  # some shape mismatch!
    prod = x * mask
    if dim is None:
        numer = torch.sum(prod)
        denom = EPS + torch.sum(mask)
    else:
        numer = torch.sum(prod, dim=dim, keepdim=keepdim)
        denom = EPS + torch.sum(mask, dim=dim, keepdim=keepdim)

    mean = numer / denom
    return mean

def balanced_ce_loss(pred, gt, valid=None):
    """Balanced cross entropy loss.
    pred: predicted scores
    gt: binary ground truth
    valid: validity mask
    """
    # pred and gt are the same shape
    for a, b in zip(pred.size(), gt.size()):
        assert a == b  # some shape mismatch!
    if valid is not None:
        for a, b in zip(pred.size(), valid.size()):
            assert a == b  # some shape mismatch!
    else:
        valid = torch.ones_like(gt)

    pos = (gt > 0.95).float()
    neg = (gt < 0.05).float()

    label = pos * 2.0 - 1.0
    a = -label * pred
    b = F.relu(a)
    loss = b + torch.log(torch.exp(-b) + torch.exp(a - b))

    pos_loss = reduce_masked_mean(loss, pos * valid)
    neg_loss = reduce_masked_mean(loss, neg * valid)

    balanced_loss = pos_loss + neg_loss

    return balanced_loss, loss

def sequence_loss(flow_preds, flow_gt, vis, valids, gamma=0.8, vis_aware=False, huber=False, delta=10, vis_aware_w=0.1, **kwargs):
    """Loss function defined over sequence of flow predictions"""
    B, S, N, D = flow_gt.shape
    assert D == 2
    B, S1, N = vis.shape
    B, S2, N = valids.shape
    assert S == S1
    assert S == S2
    n_predictions = len(flow_preds)
    flow_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        flow_pred = flow_preds[i]

        i_loss = (flow_pred - flow_gt).abs()  # B, S, N, 2
        i_loss = torch.mean(i_loss, dim=3) # B, S, N

        # Combine valids and vis for per-frame valid masking.
        combined_mask = torch.logical_and(valids, vis)
        
        # valids * vis.float() # B, S, N

        # vis_aware weighting.  Apply BEFORE reduce_masked_mean
        
        if vis_aware:
            combined_mask = combined_mask.float() * (1.0 + vis_aware_w)  # Add, don't add to the mask itself.
            # combined_mask = torch.clamp(combined_mask, 0.0, 1.0) # No need to clamp.
            # Apply the mask *before* taking the mean.
            # i_loss = i_loss * combined_mask
            # flow_loss += i_weight * i_loss.mean()
            flow_loss += i_weight * reduce_masked_mean(i_loss, combined_mask)
        else:
            if combined_mask.numel() > 10:
                # flow_loss += i_weight * i_loss.mean()
                i_loss = i_loss[combined_mask]
                flow_loss += i_weight * i_loss.mean()
            else:
                flow_loss += 0

        # # Handle the case where no points are valid.
        # if combined_mask.sum() > 0:
        #     flow_loss += i_weight * reduce_masked_mean(i_loss, combined_mask)  # Pass combined_mask
        # else:  No valid points, so this term contributes 0 to the loss.
        #     flow_loss += 0.  (This is implicit)

    # Avoid division by zero if n_predictions is 0 (though it shouldn't be).
    if n_predictions > 0:
        flow_loss = flow_loss / n_predictions

    return flow_loss

class TrackHead(nn.Module):
    """
    Track head that uses DPT/Match head to process tokens and BaseTrackerPredictor for tracking.
    """
    def __init__(self,
                 dim_in,
                 patch_size=16,
                 features=128,
                 feature_extractor_type="dpt",  # or "match"
                 train_query_points=128,
                 feature_extractor_kwargs={},
                 tracker_kwargs={},
                 loss_kwargs={},
                 iters=4,
                 use_base_tracker_v2=False,
                 predict_conf=False,
                 random_query_points = None,
                 **kwargs):
        super().__init__()
        
        self.patch_size = patch_size
        self.feature_extractor_type = feature_extractor_type
        self.train_query_points = train_query_points
        self.random_query_points = random_query_points
        
        # Initialize feature extractor (DPT or Match head)
        if feature_extractor_type == "dpt":
            self.feature_extractor = DPTHead(
                dim_in=dim_in,
                patch_size=patch_size,
                features=features,
                feature_only=True,  # Only output features, no activation
                **feature_extractor_kwargs
            )
        elif feature_extractor_type == "match":
            raise NotImplementedError("Match head is not implemented for track head")
            self.feature_extractor = MatchHead(
                dim_in=dim_in,
                patch_size=patch_size,
                features=features,
                **feature_extractor_kwargs
            )
        else:
            raise ValueError(f"Unknown feature_extractor_type: {feature_extractor_type}")
            
        # Initialize tracker
        if use_base_tracker_v2:
            self.tracker = BaseTrackerPredictorV2(
                latent_dim=features,  # Match the output_dim of feature extractor
                predict_conf=predict_conf,
                **tracker_kwargs
            )
        else:
            self.tracker = BaseTrackerPredictor(
                latent_dim=features,  # Match the output_dim of feature extractor
                predict_conf=predict_conf,
                **tracker_kwargs
            )
        
        self.loss_kwargs = loss_kwargs
        self.iters = iters
        

    def _compute_losses(self, coord_preds, vis_scores, conf_scores, batch):
        """Compute tracking losses using sequence_loss"""
        gt_tracks = batch["tracks"]  # B, S, N, 2
        gt_track_vis_mask = batch["track_vis_mask"]  # B, S, N
        
        # if self.training and hasattr(self, "train_query_points"):
        train_query_points = coord_preds[-1].shape[2]
        gt_tracks = gt_tracks[:, :, :train_query_points]
        gt_track_vis_mask = gt_track_vis_mask[:, :, :train_query_points]
    
        # Create validity mask that filters out tracks not visible in first frame
        valids = torch.ones_like(gt_track_vis_mask)
        mask = gt_track_vis_mask[:, 0, :] == True
        valids = valids * mask.unsqueeze(1)
                
        # Compute tracking loss using sequence_loss
        track_loss = sequence_loss(
            flow_preds=coord_preds,
            flow_gt=gt_tracks,
            vis=gt_track_vis_mask,
            valids=valids,
            **self.loss_kwargs
        )
        
        vis_loss = F.binary_cross_entropy_with_logits(vis_scores[valids], gt_track_vis_mask[valids].float())
        # within 3 pixels
        if conf_scores is not None:
            gt_conf_mask = (gt_tracks - coord_preds[-1]).norm(dim=-1) < 3
            conf_loss = F.binary_cross_entropy_with_logits(conf_scores[valids], gt_conf_mask[valids].float())
        else:
            conf_loss = 0
        
        return track_loss, vis_loss, conf_loss

    def forward(self, aggregated_tokens_list, batch, patch_start_idx):
        B, S, _, H, W = batch["images"].shape
        
        gt_tracks = batch["tracks"] # B, S, N, 2
        # gt_track_vis_mask = batch["track_vis_mask"] # B, S, N

        # Extract features using DPT/Match head
        if self.feature_extractor_type == "dpt":
            feature_maps = self.feature_extractor(aggregated_tokens_list, batch, patch_start_idx)
        else:  # match head
            feature_maps = self.feature_extractor(aggregated_tokens_list, batch, patch_start_idx)["descriptor"]
        
        feature_maps = feature_maps.view(B, S, *feature_maps.shape[1:]).clone()
        # Get query points from batch
        
        query_points = gt_tracks[:, 0]  # Use first frame's points as query
        
        if self.training:
            if self.random_query_points is not None:
                min_val = self.random_query_points[0]
                max_val = self.random_query_points[1]
                mu = max_val # Mean centered at the upper bound
                sigma = (max_val - min_val) / 2.71 # Standard deviation, exp
                train_query_points = int(random.gauss(mu, sigma))
                train_query_points = max(min(train_query_points, max_val), min_val) # Clamp to ensure value is within range
            else:
                train_query_points = self.train_query_points
            query_points = query_points[:, :train_query_points]
            
        # Predict tracks using BaseTrackerPredictor
        # coord_preds: a list of B, S, N, 2
        # vis_scores: B, S, N
        coord_preds, vis_scores, conf_scores = self.tracker(
            query_points=query_points,
            fmaps=feature_maps,
            iters=self.iters,  
        )
        
        # Calculate losses if in training mode
        track_loss, vis_loss, conf_loss = self._compute_losses(coord_preds, vis_scores, conf_scores, batch)
        
        loss_dict = {
            "loss_track": track_loss,
            "loss_vis": vis_loss,
            "loss_track_conf": conf_loss,
            "last_track_pred": coord_preds[-1],
        }
        return loss_dict
    
    