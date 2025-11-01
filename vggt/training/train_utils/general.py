# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import os
import math
import random
import numpy as np
from typing import Union, Optional
import logging
from iopath.common.file_io import g_pathmgr
import torch.distributed as dist
from pathlib import Path
from typing import Dict, Iterable, List



from collections import defaultdict
from dataclasses import fields, is_dataclass
from typing import Any, Mapping, Protocol, runtime_checkable




def check_and_fix_inf_nan(input_tensor, loss_name="default", hard_max=100):
    """
    Checks if 'input_tensor' contains inf or nan values and clamps extreme values.
    
    Args:
        input_tensor (torch.Tensor): The loss tensor to check and fix.
        loss_name (str): Name of the loss (for diagnostic prints).
        hard_max (float, optional): Maximum absolute value allowed. Values outside 
                                  [-hard_max, hard_max] will be clamped. If None, 
                                  no clamping is performed. Defaults to 100.
    """
    if input_tensor is None:
        return input_tensor
    
    # Check for inf/nan values
    has_inf_nan = torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any()
    if has_inf_nan:
        logging.warning(f"Tensor {loss_name} contains inf or nan values. Replacing with zeros.")
        input_tensor = torch.where(
            torch.isnan(input_tensor) | torch.isinf(input_tensor),
            torch.zeros_like(input_tensor),
            input_tensor
        )

    # Apply hard clamping if specified
    if hard_max is not None:
        input_tensor = torch.clamp(input_tensor, min=-hard_max, max=hard_max)

    return input_tensor


def get_resume_checkpoint(checkpoint_save_dir):
    if not g_pathmgr.isdir(checkpoint_save_dir):
        return None
    ckpt_file = os.path.join(checkpoint_save_dir, "checkpoint.pt")
    if not g_pathmgr.isfile(ckpt_file):
        return None

    return ckpt_file

class DurationMeter:
    def __init__(self, name, device, fmt=":f"):
        self.name = name
        self.device = device
        self.fmt = fmt
        self.val = 0

    def reset(self):
        self.val = 0

    def update(self, val):
        self.val = val

    def add(self, val):
        self.val += val

    def __str__(self):
        return f"{self.name}: {human_readable_time(self.val)}"


def human_readable_time(time_seconds):
    time = int(time_seconds)
    minutes, seconds = divmod(time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    return f"{days:02}d {hours:02}h {minutes:02}m"



class ProgressMeter:
    def __init__(self, num_batches, meters, real_meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.real_meters = real_meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        entries += [
            " | ".join(
                [
                    f"{os.path.join(name, subname)}: {val:.4f}"
                    for subname, val in meter.compute().items()
                ]
            )
            for name, meter in self.real_meters.items()
        ]
        logging.info(" | ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"



@runtime_checkable
class _CopyableData(Protocol):
    def to(self, device: torch.device, *args: Any, **kwargs: Any):
        """Copy data to the specified device"""
        ...


def _is_named_tuple(x) -> bool:
    return isinstance(x, tuple) and hasattr(x, "_asdict") and hasattr(x, "_fields")


def copy_data_to_device(data, device: torch.device, *args: Any, **kwargs: Any):
    """Function that recursively copies data to a torch.device.

    Args:
        data: The data to copy to device
        device: The device to which the data should be copied
        args: positional arguments that will be passed to the `to` call
        kwargs: keyword arguments that will be passed to the `to` call

    Returns:
        The data on the correct device
    """

    if _is_named_tuple(data):
        return type(data)(
            **copy_data_to_device(data._asdict(), device, *args, **kwargs)
        )
    elif isinstance(data, (list, tuple)):
        return type(data)(copy_data_to_device(e, device, *args, **kwargs) for e in data)
    elif isinstance(data, defaultdict):
        return type(data)(
            data.default_factory,
            {
                k: copy_data_to_device(v, device, *args, **kwargs)
                for k, v in data.items()
            },
        )
    elif isinstance(data, Mapping) and not is_dataclass(data):  # handing FrameData-like things
        return type(data)(
            {
                k: copy_data_to_device(v, device, *args, **kwargs)
                for k, v in data.items()
            }
        )
    elif is_dataclass(data) and not isinstance(data, type):
        new_data_class = type(data)(
            **{
                field.name: copy_data_to_device(
                    getattr(data, field.name), device, *args, **kwargs
                )
                for field in fields(data)
                if field.init
            }
        )
        for field in fields(data):
            if not field.init:
                setattr(
                    new_data_class,
                    field.name,
                    copy_data_to_device(
                        getattr(data, field.name), device, *args, **kwargs
                    ),
                )
        return new_data_class
    elif isinstance(data, _CopyableData):
        return data.to(device, *args, **kwargs)
    return data



def safe_makedirs(path: str):
    if not path:
        logging.warning("safe_makedirs called with an empty path. No operation performed.")
        return False

    try:
        os.makedirs(path, exist_ok=True)
        return True
    except OSError as e:
        logging.error(f"Failed to create directory '{path}'. Reason: {e}")
        raise
    except Exception as e:
        # Catch any other unexpected errors.
        logging.error(f"An unexpected error occurred while creating directory '{path}'. Reason: {e}")
        raise



def set_seeds(seed_value, max_epochs, dist_rank):
    """
    Set the python random, numpy and torch seed for each gpu. Also set the CUDA
    seeds if the CUDA is available. This ensures deterministic nature of the training.
    """
    seed_value = (seed_value + dist_rank) * max_epochs
    logging.info(f"GPU SEED: {seed_value}")
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # for multi-GPU




def log_env_variables():
    env_keys = sorted(list(os.environ.keys()))
    st = ""
    for k in env_keys:
        v = os.environ[k]
        st += f"{k}={v}\n"
    logging.info("Logging ENV_VARIABLES")
    logging.info(st)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True



class AverageMeter:
    """Computes and stores the average and current value.
    Args:
        name (str): Name of the metric being tracked
        device (torch.device, optional): Device for tensor operations. Defaults to None.
        fmt (str): Format string for displaying values. Defaults to ":f"
    """

    def __init__(self, name: str, device: Optional[torch.device] = None, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.device = device
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self._allow_updates = True

    def update(self, val, n=1):
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0

    def __str__(self) -> str:
        """String representation showing current and average values."""
        fmtstr = "{name}: {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    @property
    def value(self) -> float:
        """Get the current value."""
        return self.val

    @property
    def average(self) -> float:
        """Get the running average."""
        return self.avg

#################


_UNITS = ('', ' K', ' M', ' B', ' T')          # U+202F = thin-space for nicer look

def pretty_int(n: int) -> str:
    """Abbreviate a non-negative integer (0 → 0, 12_345 → '12.3 K')."""
    assert n >= 0, 'pretty_int() expects a non-negative int'
    if n < 1_000:
        return f'{n:,}'
    exp = int(math.log10(n) // 3)        # group of 3 digits
    exp = min(exp, len(_UNITS) - 1)      # cap at trillions
    value = n / 10 ** (3 * exp)
    return f'{value:.1f}'.rstrip('0').rstrip('.') + _UNITS[exp]


def model_summary(model: torch.nn.Module,
                  *,
                  log_file = None,
                  prefix: str = '') -> None:
    """
    Print / save a compact parameter summary.

    Args
    ----
    model      : The PyTorch nn.Module to inspect.
    log_file   : Optional path – if given, the full `str(model)` and per-parameter
                 lists are written there (three separate *.txt files).
    prefix     : Optional string printed at the beginning of every log line
                 (handy when several models share the same stdout).
    """
    if get_rank():          # only rank-0 prints
        return

    # --- counts -------------------------------------------------------------
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    frozen    = total - trainable

    print(prefix + '='*60)
    print(prefix + f'Model type : {model.__class__.__name__}')
    print(prefix + f'Total      : {pretty_int(total)} parameters')
    print(prefix + f'  trainable: {pretty_int(trainable)}')
    print(prefix + f'  frozen   : {pretty_int(frozen)}')
    print(prefix + '='*60)

    # --- optional file dump -------------------------------------------------
    if log_file is None:
        return

    log_file = Path(log_file)
    log_file.write_text(str(model))                      # full architecture

    # two extra detailed lists
    def _dump(names: Iterable[str], fname: str):
        """Write a formatted per-parameter list to *log_file.with_name(fname)*."""
        with open(log_file.with_name(fname), 'w') as f:
            for n in names:
                p = dict(model.named_parameters())[n]
                shape = str(tuple(p.shape))
                f.write(f'{n:<60s} {shape:<20} {p.numel()}\n')

    named = dict(model.named_parameters())
    _dump([n for n,p in named.items() if p.requires_grad],  'trainable.txt')
    _dump([n for n,p in named.items() if not p.requires_grad], 'frozen.txt')


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


