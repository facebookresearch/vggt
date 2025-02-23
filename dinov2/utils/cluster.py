# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from enum import Enum
import os
from pathlib import Path
from typing import Any, Dict, Optional


class ClusterType(Enum):
    AWS = "aws"
    FAIR = "fair"
    RSC = "rsc"


def _guess_cluster_type() -> ClusterType:
    uname = os.uname()
    if uname.sysname == "Linux":
        if uname.release.endswith("-aws"):
            # Linux kernel versions on AWS instances are of the form "5.4.0-1051-aws"
            return ClusterType.AWS
        elif uname.nodename.startswith("rsc"):
            # Linux kernel versions on RSC instances are standard ones but hostnames start with "rsc"
            return ClusterType.RSC

    return ClusterType.FAIR


def get_cluster_type(cluster_type: Optional[ClusterType] = None) -> Optional[ClusterType]:
    if cluster_type is None:
        return _guess_cluster_type()

    return cluster_type


def get_checkpoint_path(cluster_type: Optional[ClusterType] = None) -> Optional[Path]:
    cluster_type = get_cluster_type(cluster_type)
    if cluster_type is None:
        return None

    CHECKPOINT_DIRNAMES = {
        ClusterType.AWS: "checkpoints",
        ClusterType.FAIR: "checkpoint",
        ClusterType.RSC: "checkpoint/dino",
    }
    return Path("/") / CHECKPOINT_DIRNAMES[cluster_type]


def get_user_checkpoint_path(cluster_type: Optional[ClusterType] = None) -> Optional[Path]:
    checkpoint_path = get_checkpoint_path(cluster_type)
    if checkpoint_path is None:
        return None

    username = os.environ.get("USER")
    assert username is not None
    return checkpoint_path / username


def get_slurm_partition(cluster_type: Optional[ClusterType] = None) -> Optional[str]:
    cluster_type = get_cluster_type(cluster_type)
    if cluster_type is None:
        return None

    SLURM_PARTITIONS = {
        ClusterType.AWS: "learnlab",
        ClusterType.FAIR: "learnlab",
        ClusterType.RSC: "learn",
    }
    return SLURM_PARTITIONS[cluster_type]


def get_slurm_executor_parameters(
    nodes: int, num_gpus_per_node: int, cluster_type: Optional[ClusterType] = None, **kwargs
) -> Dict[str, Any]:
    # create default parameters
    params = {
        "mem_gb": 0,  # Requests all memory on a node, see https://slurm.schedmd.com/sbatch.html
        "gpus_per_node": num_gpus_per_node,
        "tasks_per_node": num_gpus_per_node,  # one task per GPU
        "cpus_per_task": 10,
        "nodes": nodes,
        "slurm_partition": get_slurm_partition(cluster_type),
    }
    # apply cluster-specific adjustments
    cluster_type = get_cluster_type(cluster_type)
    if cluster_type == ClusterType.AWS:
        params["cpus_per_task"] = 12
        del params["mem_gb"]
    elif cluster_type == ClusterType.RSC:
        params["cpus_per_task"] = 12
    # set additional parameters / apply overrides
    params.update(kwargs)
    return params
