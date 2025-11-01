# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for distributed training and deterministic seed generation.
This module provides functions for working with PyTorch's distributed
training capabilities and ensuring reproducible data loading.
"""

import os
import torch
import random
import numpy as np

import torch.distributed as dist
from functools import partial


def is_dist_avail_and_initialized():
    """
    Check if distributed training is available and initialized.
    
    Returns:
        bool: True if distributed training is available and initialized, False otherwise.
    """
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    """
    Get the rank of the current process in distributed training.
    
    Returns:
        int: The rank of the current process, or 0 if distributed training is not initialized.
    """
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def get_world_size():
    """
    Get the total number of processes in distributed training.
    
    Returns:
        int: The world size, or 1 if distributed training is not initialized.
    """
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def default_worker_init_fn(worker_id, num_workers, epoch, seed=0):
    """
    Default function to initialize random seeds for dataloader workers.
    
    Ensures that each worker across different ranks, epochs, and world sizes
    gets a unique random seed for reproducibility.
    
    Args:
        worker_id (int): ID of the dataloader worker.
        num_workers (int): Total number of dataloader workers.
        epoch (int): Current training epoch.
        seed (int, optional): Base seed for randomization. Defaults to 0.
    """
    rank = get_rank()
    world_size = get_world_size()
    distributed_rank = int(os.environ.get("RANK", None))

    # Use prime numbers for better distribution
    RANK_MULTIPLIER = 1
    WORKER_MULTIPLIER = 1
    WORLD_MULTIPLIER = 1
    EPOCH_MULTIPLIER = 12345
    DISTRIBUTED_RANK_MULTIPLIER = 1042
    
    worker_seed = (
        rank * num_workers * RANK_MULTIPLIER + 
        worker_id * WORKER_MULTIPLIER + 
        seed + 
        world_size * WORLD_MULTIPLIER + 
        epoch * EPOCH_MULTIPLIER
        + distributed_rank * DISTRIBUTED_RANK_MULTIPLIER
    )

    print(f"Rank: {rank}, World size: {world_size}, Distributed rank: {distributed_rank}")
    print(f"Worker seed: {worker_seed}")
    
    
    torch.random.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    return

def get_worker_init_fn(seed, num_workers, epoch, worker_init_fn=None):
    """
    Get a worker initialization function for dataloaders.
    
    Args:
        seed (int): Base seed for randomization.
        num_workers (int): Number of dataloader workers.
        epoch (int): Current training epoch.
        worker_init_fn (callable, optional): Custom worker initialization function.
            If provided, this will be returned instead of the default one.
            
    Returns:
        callable: A worker initialization function to use with DataLoader.
    """
    if worker_init_fn is not None:
        return worker_init_fn

    return partial(
        default_worker_init_fn,
        num_workers=num_workers,
        epoch=epoch,
        seed=seed,
    )
