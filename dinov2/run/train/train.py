# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import os
import sys

from dinov2.logging import setup_logging
from dinov2.train import get_args_parser as get_train_args_parser
from dinov2.run.submit import get_args_parser, submit_jobs


logger = logging.getLogger("dinov2")


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        from dinov2.train import main as train_main

        self._setup_args()
        train_main(self.args)

    def checkpoint(self):
        import submitit

        logger.info(f"Requeuing {self.args}")
        empty = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty)

    def _setup_args(self):
        import submitit

        job_env = submitit.JobEnvironment()
        self.args.output_dir = self.args.output_dir.replace("%j", str(job_env.job_id))
        logger.info(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")
        logger.info(f"Args: {self.args}")


def main():
    description = "Submitit launcher for DINOv2 training"
    train_args_parser = get_train_args_parser(add_help=False)
    parents = [train_args_parser]
    args_parser = get_args_parser(description=description, parents=parents)
    args = args_parser.parse_args()

    setup_logging()

    assert os.path.exists(args.config_file), "Configuration file does not exist!"
    submit_jobs(Trainer, args, name="dinov2:train")
    return 0


if __name__ == "__main__":
    sys.exit(main())
