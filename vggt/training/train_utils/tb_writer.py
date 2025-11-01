# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import atexit
import logging
import uuid
from typing import Any, Dict, Optional, Union

import torch
from torch.utils.tensorboard import SummaryWriter

from .distributed import get_machine_local_and_dist_rank


class TensorBoardLogger:
    """A wrapper around TensorBoard SummaryWriter with distributed training support.

    This logger only writes from rank 0 in distributed settings to avoid conflicts.
    Automatically handles cleanup on exit.
    """

    def __init__(
        self,
        path: str,
        *args: Any,
        filename_suffix: Optional[str] = None,
        summary_writer_method: Any = SummaryWriter,
        **kwargs: Any,
    ) -> None:
        """Initialize TensorBoard logger.

        Args:
            path: Directory path where TensorBoard logs will be stored
            filename_suffix: Optional suffix for log filename. If None, uses random UUID
            summary_writer_method: SummaryWriter class or compatible alternative
            *args, **kwargs: Additional arguments passed to SummaryWriter
        """
        self._writer: Optional[SummaryWriter] = None
        _, self._rank = get_machine_local_and_dist_rank()
        self._path: str = path
        if self._rank == 0:
            logging.info(
                f"TensorBoard SummaryWriter instantiated. Files will be stored in: {path}"
            )
            self._writer = summary_writer_method(
                log_dir=path,
                *args,
                filename_suffix=filename_suffix or str(uuid.uuid4()),
                **kwargs,
            )
        else:
            logging.debug(
                f"Not logging on this process because rank {self._rank} != 0"
            )

        atexit.register(self.close)

    @property
    def writer(self) -> Optional[SummaryWriter]:
        """Get the underlying SummaryWriter instance."""
        return self._writer

    @property
    def path(self) -> str:
        """Get the log directory path."""
        return self._path

    def flush(self) -> None:
        """Write pending logs to disk."""
        if self._writer:
            self._writer.flush()

    def close(self) -> None:
        """Close writer and flush pending logs to disk.

        Logs cannot be written after close() is called.
        """
        if self._writer:
            self._writer.close()
            self._writer = None

    def log_dict(self, payload: Dict[str, Any], step: int) -> None:
        """Log multiple scalar values to TensorBoard.

        Args:
            payload: Dictionary mapping tag names to scalar values
            step: Step value to record
        """
        if not self._writer:
            return

        for key, value in payload.items():
            self.log(key, value, step)

    def log(self, name: str, data: Any, step: int) -> None:
        """Log scalar data to TensorBoard.

        Args:
            name: Tag name used to group scalars
            data: Scalar data to log (float/int/Tensor)
            step: Step value to record
        """
        if not self._writer:
            return

        self._writer.add_scalar(name, data, global_step=step, new_style=True)

    def log_visuals(
        self,
        name: str,
        data: Union[torch.Tensor, Any],
        step: int,
        fps: int = 4
    ) -> None:
        """Log image or video data to TensorBoard.

        Args:
            name: Tag name used to group visuals
            data: Image tensor (3D) or video tensor (5D)
            step: Step value to record
            fps: Frames per second for video data

        Raises:
            ValueError: If data dimensions are not supported (must be 3D or 5D)
        """
        if not self._writer:
            return

        if data.ndim == 3:
            self._writer.add_image(name, data, global_step=step)
        elif data.ndim == 5:
            self._writer.add_video(name, data, global_step=step, fps=fps)
        else:
            raise ValueError(
                f"Unsupported data dimensions: {data.ndim}. "
                "Expected 3D for images or 5D for videos."
            )
