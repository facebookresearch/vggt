# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
# 
# --------------------------------------------------------
# Criterion to train CroCo
# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import torch

class MaskedMSE(torch.nn.Module):

    def __init__(self, norm_pix_loss=False, masked=True):
        """
            norm_pix_loss: normalize each patch by their pixel mean and variance
            masked: compute loss over the masked patches only 
        """
        super().__init__()
        self.norm_pix_loss = norm_pix_loss
        self.masked = masked 
        
    def forward(self, pred, mask, target):
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
            
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        if self.masked:
            loss = (loss * mask).sum() / mask.sum()  # mean loss on masked patches
        else:
            loss = loss.mean()  # mean loss
        return loss