# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from abc import ABCMeta, abstractmethod

import torch

from ..builder import MASK_ASSIGNERS, build_match_cost

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


class AssignResult(metaclass=ABCMeta):
    """Collection of assign results."""

    def __init__(self, num_gts, gt_inds, labels):
        self.num_gts = num_gts
        self.gt_inds = gt_inds
        self.labels = labels

    @property
    def info(self):
        info = {
            "num_gts": self.num_gts,
            "gt_inds": self.gt_inds,
            "labels": self.labels,
        }
        return info


class BaseAssigner(metaclass=ABCMeta):
    """Base assigner that assigns boxes to ground truth boxes."""

    @abstractmethod
    def assign(self, masks, gt_masks, gt_masks_ignore=None, gt_labels=None):
        """Assign boxes to either a ground truth boxes or a negative boxes."""
        pass


@MASK_ASSIGNERS.register_module()
class MaskHungarianAssigner(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth for
    mask.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        cls_cost (obj:`mmcv.ConfigDict`|dict): Classification cost config.
        mask_cost (obj:`mmcv.ConfigDict`|dict): Mask cost config.
        dice_cost (obj:`mmcv.ConfigDict`|dict): Dice cost config.
    """

    def __init__(
        self,
        cls_cost=dict(type="ClassificationCost", weight=1.0),
        dice_cost=dict(type="DiceCost", weight=1.0),
        mask_cost=dict(type="MaskFocalCost", weight=1.0),
    ):
        self.cls_cost = build_match_cost(cls_cost)
        self.dice_cost = build_match_cost(dice_cost)
        self.mask_cost = build_match_cost(mask_cost)

    def assign(self, cls_pred, mask_pred, gt_labels, gt_masks, img_meta, gt_masks_ignore=None, eps=1e-7):
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            mask_pred (Tensor): Predicted mask, shape [num_query, h, w]
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_masks (Tensor): Ground truth mask, shape [num_gt, h, w].
            gt_labels (Tensor): Label of `gt_masks`, shape (num_gt,).
            img_meta (dict): Meta information for current image.
            gt_masks_ignore (Tensor, optional): Ground truth masks that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert gt_masks_ignore is None, "Only case when gt_masks_ignore is None is supported."
        num_gts, num_queries = gt_labels.shape[0], cls_pred.shape[0]

        # 1. assign -1 by default
        assigned_gt_inds = cls_pred.new_full((num_queries,), -1, dtype=torch.long)
        assigned_labels = cls_pred.new_full((num_queries,), -1, dtype=torch.long)
        if num_gts == 0 or num_queries == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(num_gts, assigned_gt_inds, labels=assigned_labels)

        # 2. compute the weighted costs
        # classification and maskcost.
        if self.cls_cost.weight != 0 and cls_pred is not None:
            cls_cost = self.cls_cost(cls_pred, gt_labels)
        else:
            cls_cost = 0

        if self.mask_cost.weight != 0:
            # mask_pred shape = [nq, h, w]
            # gt_mask shape = [ng, h, w]
            # mask_cost shape = [nq, ng]
            mask_cost = self.mask_cost(mask_pred, gt_masks)
        else:
            mask_cost = 0

        if self.dice_cost.weight != 0:
            dice_cost = self.dice_cost(mask_pred, gt_masks)
        else:
            dice_cost = 0
        cost = cls_cost + mask_cost + dice_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" ' "to install scipy first.")

        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(cls_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(cls_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(num_gts, assigned_gt_inds, labels=assigned_labels)
