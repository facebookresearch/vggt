# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import functools
import logging
import os
import sys
from typing import Optional

import dinov2.distributed as distributed
from .helpers import MetricLogger, SmoothedValue


# So that calling _configure_logger multiple times won't add many handlers
@functools.lru_cache()
def _configure_logger(
    name: Optional[str] = None,
    *,
    level: int = logging.DEBUG,
    output: Optional[str] = None,
):
    """
    Configure a logger.

    Adapted from Detectron2.

    Args:
        name: The name of the logger to configure.
        level: The logging level to use.
        output: A file name or a directory to save log. If None, will not save log file.
            If ends with ".txt" or ".log", assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.

    Returns:
        The configured logger.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Loosely match Google glog format:
    #   [IWEF]yyyymmdd hh:mm:ss.uuuuuu threadid file:line] msg
    # but use a shorter timestamp and include the logger name:
    #   [IWEF]yyyymmdd hh:mm:ss logger threadid file:line] msg
    fmt_prefix = "%(levelname).1s%(asctime)s %(process)s %(name)s %(filename)s:%(lineno)s] "
    fmt_message = "%(message)s"
    fmt = fmt_prefix + fmt_message
    datefmt = "%Y%m%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    # stdout logging for main worker only
    if distributed.is_main_process():
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # file logging for all workers
    if output:
        if os.path.splitext(output)[-1] in (".txt", ".log"):
            filename = output
        else:
            filename = os.path.join(output, "logs", "log.txt")

        if not distributed.is_main_process():
            global_rank = distributed.get_global_rank()
            filename = filename + ".rank{}".format(global_rank)

        os.makedirs(os.path.dirname(filename), exist_ok=True)

        handler = logging.StreamHandler(open(filename, "a"))
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def setup_logging(
    output: Optional[str] = None,
    *,
    name: Optional[str] = None,
    level: int = logging.DEBUG,
    capture_warnings: bool = True,
) -> None:
    """
    Setup logging.

    Args:
        output: A file name or a directory to save log files. If None, log
            files will not be saved. If output ends with ".txt" or ".log", it
            is assumed to be a file name.
            Otherwise, logs will be saved to `output/log.txt`.
        name: The name of the logger to configure, by default the root logger.
        level: The logging level to use.
        capture_warnings: Whether warnings should be captured as logs.
    """
    logging.captureWarnings(capture_warnings)
    _configure_logger(name, level=level, output=output)
