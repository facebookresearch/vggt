# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from typing import Union, Optional


class GradientClipper:
    """
    Gradient clipping utils that works for both FSDP and DDP with support for different
    clipping configurations for different parts of the model.
    """
    def __init__(self, configs, *args, **kwargs):
        """
        Args:
            configs: List of dictionaries, each containing:
                - module_name: str or list of str, module names to apply clipping to
                - max_norm: float, maximum norm for gradient clipping
                - norm_type: int, type of norm (default: 2)
        """
        self.configs = []
        self.params_to_clip_by_config = None
        self.is_initialized = False
        
        for config in configs:
            module_names = config['module_name']
            if isinstance(module_names, str):
                module_names = [module_names]
            
            self.configs.append({
                'module_names': module_names,
                'max_norm': float(config['max_norm']) if config['max_norm'] is not None else None,
                'norm_type': config.get('norm_type', 2)
            })

    def setup_clipping(self, model: nn.Module) -> None:
        """
        Set up gradient clipping by finding all parameters that should be clipped
        based on module names and validating that all parameters are covered.
        
        This should be called once at the beginning of training.
        
        Args:
            model: The model to set up gradient clipping for
        """
        # First, collect all parameters that should be clipped based on module names
        params_to_clip_by_config = []
        all_clipped_params = set()
        
        for config in self.configs:
            current_config_params = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    for module_name in config['module_names']:
                        if module_name in name:
                            current_config_params.append(param)
                            all_clipped_params.add(param)
                            break
            params_to_clip_by_config.append((config, current_config_params))

        # Check for remaining parameters
        remaining_params = []
        for name, param in model.named_parameters():
            if param.requires_grad and param not in all_clipped_params:
                remaining_params.append(param)

        if len(remaining_params) > 0:
            print(f"Found {len(remaining_params)} parameters that won't be clipped")
            print(remaining_params)
            raise ValueError("Some parameters are not configured for gradient clipping")
        
        # Store the computed parameters
        self.params_to_clip_by_config = params_to_clip_by_config
        self.is_initialized = True

    def __call__(self, model: nn.Module) -> Optional[torch.Tensor]:
        """
        Perform gradient clipping using the pre-computed parameter groups.
        
        Args:
            model: The model (not used, kept for backward compatibility)
            
        Returns:
            Dictionary of gradient norms for each configuration
        """
        if not self.is_initialized:
            raise RuntimeError("GradientClipper must be initialized with setup_clipping() before use")
        
        grad_norms = {}
        for config, params_to_clip in self.params_to_clip_by_config:
            if not params_to_clip or config['max_norm'] is None:
                continue

            grad_norm = nn.utils.clip_grad_norm_(
                params_to_clip,
                max_norm=config['max_norm'],
                norm_type=config['norm_type']
            )

            if grad_norm is None:
                continue
            
            grad_norms[",".join(config['module_names'])] = grad_norm.item()

        return grad_norms
