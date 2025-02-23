# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from mmcv.utils import Registry

TRANSFORMER = Registry("Transformer")
MASK_ASSIGNERS = Registry("mask_assigner")
MATCH_COST = Registry("match_cost")


def build_match_cost(cfg):
    """Build Match Cost."""
    return MATCH_COST.build(cfg)


def build_assigner(cfg):
    """Build Assigner."""
    return MASK_ASSIGNERS.build(cfg)


def build_transformer(cfg):
    """Build Transformer."""
    return TRANSFORMER.build(cfg)
