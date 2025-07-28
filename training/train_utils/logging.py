# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import copy
import sys
import atexit

import functools
from .general import safe_makedirs
from iopath.common.file_io import g_pathmgr


# cache the opened file object, so that different calls
# with the same file name can safely write to the same file.
@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    log_buffer_kb = 1 * 1024  # 1KB
    io = g_pathmgr.open(filename, mode="a", buffering=log_buffer_kb)
    atexit.register(io.close)
    return io



def setup_logging(
    name,
    output_dir=None,
    rank=0,
    log_level_primary="INFO",
    log_level_secondary="ERROR",
    all_ranks: bool = False,
):
    """
    Setup various logging streams: stdout and file handlers.
    For file handlers, we only setup for the master gpu.
    """
    global LOGGING_STATE
    LOGGING_STATE = copy.deepcopy(locals())

    # get the filename if we want to log to the file as well
    log_filename = None
    if output_dir:
        safe_makedirs(output_dir)
        if rank == 0:
            log_filename = f"{output_dir}/log.txt"
        elif all_ranks:
            log_filename = f"{output_dir}/log_{rank}.txt"

    logger = logging.getLogger(name)
    logger.setLevel(log_level_primary)

    # create formatter
    FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)4d: %(message)s"
    formatter = logging.Formatter(FORMAT)

    # clean up any pre-existing handlers
    for h in logger.handlers:
        logger.removeHandler(h)
    logger.root.handlers = []
    logging.root.handlers = []

    # setup the console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    if rank == 0:
        console_handler.setLevel(log_level_primary)
    else:
        console_handler.setLevel(log_level_secondary)
    logger.addHandler(console_handler)

    # we log to file as well if user wants
    if log_filename is not None:
        file_handler = logging.StreamHandler(_cached_log_stream(log_filename))
        file_handler.setLevel(log_level_primary)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logging.root = logger
