# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import warnings

from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.cnn.bricks.registry import ATTENTION as MMCV_ATTENTION
from mmcv.utils import Registry

MODELS = Registry("models", parent=MMCV_MODELS)
ATTENTION = Registry("attention", parent=MMCV_ATTENTION)


BACKBONES = MODELS
NECKS = MODELS
HEADS = MODELS
LOSSES = MODELS
DEPTHER = MODELS


def build_backbone(cfg):
    """Build backbone."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck."""
    return NECKS.build(cfg)


def build_head(cfg):
    """Build head."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_depther(cfg, train_cfg=None, test_cfg=None):
    """Build depther."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn("train_cfg and test_cfg is deprecated, " "please specify them in model", UserWarning)
    assert cfg.get("train_cfg") is None or train_cfg is None, "train_cfg specified in both outer field and model field "
    assert cfg.get("test_cfg") is None or test_cfg is None, "test_cfg specified in both outer field and model field "
    return DEPTHER.build(cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))
