# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional

import submitit

from dinov2.utils.cluster import (
    get_slurm_executor_parameters,
    get_slurm_partition,
    get_user_checkpoint_path,
)


logger = logging.getLogger("dinov2")


def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
) -> argparse.ArgumentParser:
    parents = parents or []
    slurm_partition = get_slurm_partition()
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents,
        add_help=add_help,
    )
    parser.add_argument(
        "--ngpus",
        "--gpus",
        "--gpus-per-node",
        default=8,
        type=int,
        help="Number of GPUs to request on each node",
    )
    parser.add_argument(
        "--nodes",
        "--nnodes",
        default=1,
        type=int,
        help="Number of nodes to request",
    )
    parser.add_argument(
        "--timeout",
        default=2800,
        type=int,
        help="Duration of the job",
    )
    parser.add_argument(
        "--partition",
        default=slurm_partition,
        type=str,
        help="Partition where to submit",
    )
    parser.add_argument(
        "--use-volta32",
        action="store_true",
        help="Request V100-32GB GPUs",
    )
    parser.add_argument(
        "--comment",
        default="",
        type=str,
        help="Comment to pass to scheduler, e.g. priority message",
    )
    parser.add_argument(
        "--exclude",
        default="",
        type=str,
        help="Nodes to exclude",
    )
    return parser


def get_shared_folder() -> Path:
    user_checkpoint_path = get_user_checkpoint_path()
    if user_checkpoint_path is None:
        raise RuntimeError("Path to user checkpoint cannot be determined")
    path = user_checkpoint_path / "experiments"
    path.mkdir(exist_ok=True)
    return path


def submit_jobs(task_class, args, name: str):
    if not args.output_dir:
        args.output_dir = str(get_shared_folder() / "%j")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=args.output_dir, slurm_max_num_timeout=30)

    kwargs = {}
    if args.use_volta32:
        kwargs["slurm_constraint"] = "volta32gb"
    if args.comment:
        kwargs["slurm_comment"] = args.comment
    if args.exclude:
        kwargs["slurm_exclude"] = args.exclude

    executor_params = get_slurm_executor_parameters(
        nodes=args.nodes,
        num_gpus_per_node=args.ngpus,
        timeout_min=args.timeout,  # max is 60 * 72
        slurm_signal_delay_s=120,
        slurm_partition=args.partition,
        **kwargs,
    )
    executor.update_parameters(name=name, **executor_params)

    task = task_class(args)
    job = executor.submit(task)

    logger.info(f"Submitted job_id: {job.job_id}")
    str_output_dir = os.path.abspath(args.output_dir).replace("%j", str(job.job_id))
    logger.info(f"Logs and checkpoints will be saved at: {str_output_dir}")
