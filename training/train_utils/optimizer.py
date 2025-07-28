# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import itertools
from typing import Any, Dict, List, Mapping, Iterable, Set, Tuple, Union

import hydra
import torch
import torch.nn as nn
from torch import Tensor

# -----------------------------------------------------------------------------
# Optimizer wrapper
# -----------------------------------------------------------------------------

class OptimizerWrapper:
    """Wraps a torch.optim.Optimizer and its schedulers (if any)."""

    def __init__(self, optimizer: torch.optim.Optimizer, schedulers=None) -> None:
        self.optimizer = optimizer
        self.schedulers = schedulers
        self._validate_optimizer_schedulers()
        self.step_schedulers(0.0)

    # ---------------------------------------------------------------------
    # Public API mirroring torch.optim.Optimizer
    # ---------------------------------------------------------------------

    def step(self, where: float = 1.0, closure=None):
        """Update the optimizer & its schedulers."""
        self.step_schedulers(where)
        return self.optimizer.step(closure)

    def zero_grad(self, *args, **kwargs):
        return self.optimizer.zero_grad(*args, **kwargs)

    def _validate_optimizer_schedulers(self):
        if self.schedulers is None:
            return
        for _, sched_map in enumerate(self.schedulers):
            for option, _ in sched_map.items():
                assert option in self.optimizer.defaults, (
                    f"Optimizer option {option} not found in {self.optimizer}. "
                    f"Valid options are {self.optimizer.defaults.keys()}"
                )

    def step_schedulers(self, where: float) -> None:
        if self.schedulers is None:
            return
        for i, param_group in enumerate(self.optimizer.param_groups):
            for option, scheduler in self.schedulers[i].items():
                param_group[option] = scheduler(where)


# -----------------------------------------------------------------------------
# Validation helpers
# -----------------------------------------------------------------------------


def validate_param_group_params(param_groups: List[Dict], model: nn.Module):
    """Ensure param groups are non-overlapping and include all model params."""

    for pg in param_groups:
        assert len(pg["params"]) == len(set(pg["params"]))

    parameters = [set(pg["params"]) for pg in param_groups]
    model_parameters = {p for _, p in model.named_parameters()}

    for p1, p2 in itertools.permutations(parameters, 2):
        assert p1.isdisjoint(p2), "Parameter groups should be disjoint"

    assert set.union(*parameters) == model_parameters, (
        "Parameter groups must cover ALL model parameters "
        f"(found {len(set.union(*parameters))} / {len(model_parameters)})"
    )


# -----------------------------------------------------------------------------
# Glob helpers for pattern matching
# -----------------------------------------------------------------------------

from wcmatch import fnmatch

GLOB_FLAGS = (
    fnmatch.CASE       # case-sensitive
    | fnmatch.DOTMATCH # '*' also matches '.'
    | fnmatch.EXTMATCH # extended patterns like *(foo|bar)
    | fnmatch.SPLIT    # "pat1|pat2" works out-of-the-box
)


def get_full_parameter_name(module_name: str, param_name: str) -> str:
    return param_name if module_name == "" else f"{module_name}.{param_name}"


def get_module_cls_to_param_names(model: nn.Module) -> Dict[type, Set[str]]:
    """Map each module class to the *immediate* param names it owns."""
    mapping: Dict[type, Set[str]] = {}
    for module_name, module in model.named_modules():
        module_cls = type(module)
        mapping.setdefault(module_cls, set())
        for pname, _ in module.named_parameters(recurse=False):
            mapping[module_cls].add(get_full_parameter_name(module_name, pname))
    return mapping


def unix_param_pattern_to_parameter_names(filter_param_names: Union[List[str], None],
                                           parameter_names: Set[str]) -> Set[str]:
    if filter_param_names is None:
        return set()
    allowed = []
    for pat in filter_param_names:
        matches = set(fnmatch.filter(parameter_names, pat, flags=GLOB_FLAGS))
        if not matches:
            raise AssertionError(f"Pattern {pat} matched no parameters")
        logging.info(f"Matches for param pattern [{pat}]: {matches}")
        allowed.append(matches)
    return set.union(*allowed)


def unix_module_cls_pattern_to_parameter_names(filter_module_cls_names: Union[List[str], None],
                                               module_cls_to_param_names: Dict[type, Set[str]]) -> Set[str]:
    if filter_module_cls_names is None:
        return set()
    allowed = []
    for cls_name in filter_module_cls_names:
        module_cls = hydra.utils.get_class(cls_name)
        if module_cls not in module_cls_to_param_names:
            raise AssertionError(f"Module class {cls_name} not found in model")
        params = module_cls_to_param_names[module_cls]
        if not params:
            raise AssertionError(f"Module class {cls_name} has no parameters")
        logging.info(f"Matches for module [{cls_name}]: {params}")
        allowed.append(params)
    return set.union(*allowed)


def _unix_pattern_to_parameter_names(scheduler_cfg,
                                     parameter_names: Set[str],
                                     module_cls_to_param_names: Dict[type, Set[str]]):
    if "param_names" not in scheduler_cfg and "module_cls_names" not in scheduler_cfg:
        return None
    return unix_param_pattern_to_parameter_names(
        scheduler_cfg.get("param_names"), parameter_names
    ).union(
        unix_module_cls_pattern_to_parameter_names(
            scheduler_cfg.get("module_cls_names"), module_cls_to_param_names
        )
    )


# -----------------------------------------------------------------------------
# Scheduler helpers
# -----------------------------------------------------------------------------


def set_default_parameters(scheduler_cfgs: List[dict], all_parameter_names: Set[str]):
    """Ensure exactly one scheduler per option acts as the default."""
    specified = [cfg["parameter_names"] for cfg in scheduler_cfgs if cfg["parameter_names"]]

    default_params = (
        all_parameter_names if not specified else all_parameter_names - set.union(*specified)
    )

    default_count = 0
    for cfg in scheduler_cfgs:
        if cfg["parameter_names"] is None:
            cfg["parameter_names"] = default_params
            default_count += 1
    assert default_count <= 1, "At most one default scheduler per option"

    if default_count == 0:
        scheduler_cfgs.append({"parameter_names": default_params})


def name_constraints_to_parameters(param_constraints: List[Set[str]],
                                   named_parameters: Dict[str, Tensor]) -> List[Tensor]:
    matching_names = set.intersection(*param_constraints)
    return [v for k, v in named_parameters.items() if k in matching_names]


def map_scheduler_cfgs_to_param_groups(all_scheduler_cfgs: Iterable[List[dict]],
                                       named_parameters: Dict[str, Tensor]):
    """Produce param groups & schedulers that torch.optim can consume."""
    schedulers: List[Dict[str, Any]] = []
    param_groups: List[Dict[str, List[Tensor]]] = []

    for cfgs in itertools.product(*all_scheduler_cfgs):
        param_constraints = [cfg["parameter_names"] for cfg in cfgs]
        matching = name_constraints_to_parameters(param_constraints, named_parameters)
        if not matching:
            continue  # no intersection of params for this combo
        schedulers.append({cfg["option"]: cfg["scheduler"] for cfg in cfgs if "option" in cfg})
        param_groups.append({"params": matching})

    return schedulers, param_groups


# -----------------------------------------------------------------------------
# Public factory functions
# -----------------------------------------------------------------------------


def construct_optimizer(model: nn.Module,
                        optimizer_conf: Any,
                        options_conf: Union[Mapping[str, List], None] = None,
                        param_group_modifiers_conf: Union[List, None] = None,
                        validate_param_groups: bool = True) -> OptimizerWrapper:
    """Build an OptimizerWrapper from hydra configs.

    *No* allowlist handling – we always optimize *all* model parameters.
    """

    named_parameters = dict(model.named_parameters())
    all_parameter_names = set(named_parameters.keys())
    module_cls_to_all_param_names = get_module_cls_to_param_names(model)

    # ──────────────────────────────────────────────────────────────────
    # No scheduler case – simple & fast
    # ──────────────────────────────────────────────────────────────────
    if not options_conf:
        optimizer = hydra.utils.instantiate(optimizer_conf, named_parameters.values())
        return OptimizerWrapper(optimizer)

    # ──────────────────────────────────────────────────────────────────
    # Build option-specific scheduler configs
    # ──────────────────────────────────────────────────────────────────
    scheduler_cfgs_per_option = hydra.utils.instantiate(options_conf)
    all_scheduler_cfgs: List[List[dict]] = []

    for option, cfg_list in scheduler_cfgs_per_option.items():
        for cfg in cfg_list:
            cfg.option = option  # annotate
            cfg.parameter_names = _unix_pattern_to_parameter_names(
                cfg, all_parameter_names, module_cls_to_all_param_names
            )
        set_default_parameters(cfg_list, all_parameter_names)
        all_scheduler_cfgs.append(cfg_list)

    # User-provided modifiers (rare)
    if param_group_modifiers_conf:
        for modifier in param_group_modifiers_conf:
            modifier = hydra.utils.instantiate(modifier)
            all_scheduler_cfgs = modifier(scheduler_cfgs=all_scheduler_cfgs, model=model)

    # Map scheduler cfg combos to optimizer param groups
    schedulers, param_groups = map_scheduler_cfgs_to_param_groups(
        all_scheduler_cfgs, named_parameters
    )

    if validate_param_groups:
        validate_param_group_params(param_groups, model)

    optimizer = hydra.utils.instantiate(optimizer_conf, param_groups)
    return OptimizerWrapper(optimizer, schedulers)


def construct_optimizers(model: nn.Module, optim_conf) -> Union[List[OptimizerWrapper], None]:
    """Convenience wrapper producing a *single* OptimizerWrapper list."""
    if optim_conf is None:
        return None

    optimizer = construct_optimizer(
        model,
        optim_conf.optimizer,
        optim_conf.options,
        validate_param_groups=True,
    )
    return [optimizer]
