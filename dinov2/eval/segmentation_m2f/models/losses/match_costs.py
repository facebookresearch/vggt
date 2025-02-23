# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from ..builder import MATCH_COST


@MATCH_COST.register_module()
class ClassificationCost:
    """ClsSoftmaxCost.Borrow from
    mmdet.core.bbox.match_costs.match_cost.ClassificationCost.

     Args:
         weight (int | float, optional): loss_weight

     Examples:
         >>> import torch
         >>> self = ClassificationCost()
         >>> cls_pred = torch.rand(4, 3)
         >>> gt_labels = torch.tensor([0, 1, 2])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(cls_pred, gt_labels)
         tensor([[-0.3430, -0.3525, -0.3045],
                [-0.3077, -0.2931, -0.3992],
                [-0.3664, -0.3455, -0.2881],
                [-0.3343, -0.2701, -0.3956]])
    """

    def __init__(self, weight=1.0):
        self.weight = weight

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        # Following the official DETR repo, contrary to the loss that
        # NLL is used, we approximate it in 1 - cls_score[gt_label].
        # The 1 is a constant that doesn't change the matching,
        # so it can be omitted.
        cls_score = cls_pred.softmax(-1)
        cls_cost = -cls_score[:, gt_labels]
        return cls_cost * self.weight


@MATCH_COST.register_module()
class DiceCost:
    """Cost of mask assignments based on dice losses.

    Args:
        weight (int | float, optional): loss_weight. Defaults to 1.
        pred_act (bool, optional): Whether to apply sigmoid to mask_pred.
            Defaults to False.
        eps (float, optional): default 1e-12.
    """

    def __init__(self, weight=1.0, pred_act=False, eps=1e-3):
        self.weight = weight
        self.pred_act = pred_act
        self.eps = eps

    def binary_mask_dice_loss(self, mask_preds, gt_masks):
        """
        Args:
            mask_preds (Tensor): Mask prediction in shape (N1, H, W).
            gt_masks (Tensor): Ground truth in shape (N2, H, W)
                store 0 or 1, 0 for negative class and 1 for
                positive class.

        Returns:
            Tensor: Dice cost matrix in shape (N1, N2).
        """
        mask_preds = mask_preds.reshape((mask_preds.shape[0], -1))
        gt_masks = gt_masks.reshape((gt_masks.shape[0], -1)).float()
        numerator = 2 * torch.einsum("nc,mc->nm", mask_preds, gt_masks)
        denominator = mask_preds.sum(-1)[:, None] + gt_masks.sum(-1)[None, :]
        loss = 1 - (numerator + self.eps) / (denominator + self.eps)
        return loss

    def __call__(self, mask_preds, gt_masks):
        """
        Args:
            mask_preds (Tensor): Mask prediction logits in shape (N1, H, W).
            gt_masks (Tensor): Ground truth in shape (N2, H, W).

        Returns:
            Tensor: Dice cost matrix in shape (N1, N2).
        """
        if self.pred_act:
            mask_preds = mask_preds.sigmoid()
        dice_cost = self.binary_mask_dice_loss(mask_preds, gt_masks)
        return dice_cost * self.weight


@MATCH_COST.register_module()
class CrossEntropyLossCost:
    """CrossEntropyLossCost.

    Args:
        weight (int | float, optional): loss weight. Defaults to 1.
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to True.
    """

    def __init__(self, weight=1.0, use_sigmoid=True):
        assert use_sigmoid, "use_sigmoid = False is not supported yet."
        self.weight = weight
        self.use_sigmoid = use_sigmoid

    def _binary_cross_entropy(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): The prediction with shape (num_query, 1, *) or
                (num_query, *).
            gt_labels (Tensor): The learning label of prediction with
                shape (num_gt, *).
        Returns:
            Tensor: Cross entropy cost matrix in shape (num_query, num_gt).
        """
        cls_pred = cls_pred.flatten(1).float()
        gt_labels = gt_labels.flatten(1).float()
        n = cls_pred.shape[1]
        pos = F.binary_cross_entropy_with_logits(cls_pred, torch.ones_like(cls_pred), reduction="none")
        neg = F.binary_cross_entropy_with_logits(cls_pred, torch.zeros_like(cls_pred), reduction="none")
        cls_cost = torch.einsum("nc,mc->nm", pos, gt_labels) + torch.einsum("nc,mc->nm", neg, 1 - gt_labels)
        cls_cost = cls_cost / n

        return cls_cost

    def __call__(self, cls_pred, gt_labels):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits.
            gt_labels (Tensor): Labels.
        Returns:
            Tensor: Cross entropy cost matrix with weight in
                shape (num_query, num_gt).
        """
        if self.use_sigmoid:
            cls_cost = self._binary_cross_entropy(cls_pred, gt_labels)
        else:
            raise NotImplementedError

        return cls_cost * self.weight
