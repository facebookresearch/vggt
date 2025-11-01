# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from wcmatch import fnmatch
from functools import wraps
from typing import List

import torch.nn as nn

# ------------------------------------------------------------
# Glob‑matching flags (behave like the Unix shell) 
# ------------------------------------------------------------
GLOB_FLAGS = (
    fnmatch.CASE       # case‑sensitive
    | fnmatch.DOTMATCH # '*' also matches '.'
    | fnmatch.EXTMATCH # extended patterns like *(foo|bar)
    | fnmatch.SPLIT    # "pat1|pat2" works out‑of‑the‑box
)


def freeze_modules(model: nn.Module, patterns: List[str], recursive: bool = True) -> nn.Module:
    """Freeze (stop training) parts of *model* whose *name* matches *patterns*.

    Parameters
    ----------
    model : nn.Module
        The complete model you are working with.
    patterns : list[str]
        Glob patterns to match sub‑module names.  Example: ``["encoder.*", "cls_head"]``
    recursive : bool, default = True
        • ``True``  → also freeze every child of a matched module.
        • ``False`` → freeze only the matched module itself.

    Returns
    -------
    nn.Module
        The same model object, now with some parts frozen.

    Example
    -------
    >>> freeze_modules(model, ["encoder.*", "decoder.layer1"], recursive=True)
    """
    matched: set[str] = set()

    for name, mod in model.named_modules():
        # does *name* match ANY user pattern?
        if any(fnmatch.fnmatch(name, p, flags=GLOB_FLAGS) for p in patterns):
            matched.add(name)
            _freeze(mod, recursive)

    _check_every_pattern_used(matched, patterns)
    return model


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def _freeze(mod: nn.Module, recursive: bool) -> None:
    """Put *mod* in eval mode and lock its parameters."""

    if recursive:
        mod.eval()            # affects the whole subtree
    else:
        mod.training = False  # only this exact module

    original_train = mod.train

    @wraps(original_train)
    def locked_train(mode: bool = True):
        if recursive:
            return original_train(False)  # ignore user's *mode*
        out = original_train(mode)        # children follow user's choice
        out.training = False              # but this module stays frozen
        return out

    mod.train = locked_train  # type: ignore[attr-defined]

    param_iter = (
        mod.parameters()              # default recurse=True
        if recursive
        else mod.parameters(recurse=False)
    )
    for p in param_iter:
        p.requires_grad = False


def _check_every_pattern_used(matched_names: set[str], patterns: List[str]):
    unused = [p for p in patterns if not any(fnmatch.fnmatch(n, p, flags=GLOB_FLAGS)
                                             for n in matched_names)]
    if unused:
        raise ValueError(f"These patterns matched nothing: {unused}")
