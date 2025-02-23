# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


from typing import Dict, Union

import numpy as np
import torch


TypeSpec = Union[str, np.dtype, torch.dtype]


_NUMPY_TO_TORCH_DTYPE: Dict[np.dtype, torch.dtype] = {
    np.dtype("bool"): torch.bool,
    np.dtype("uint8"): torch.uint8,
    np.dtype("int8"): torch.int8,
    np.dtype("int16"): torch.int16,
    np.dtype("int32"): torch.int32,
    np.dtype("int64"): torch.int64,
    np.dtype("float16"): torch.float16,
    np.dtype("float32"): torch.float32,
    np.dtype("float64"): torch.float64,
    np.dtype("complex64"): torch.complex64,
    np.dtype("complex128"): torch.complex128,
}


def as_torch_dtype(dtype: TypeSpec) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        dtype = np.dtype(dtype)
    assert isinstance(dtype, np.dtype), f"Expected an instance of nunpy dtype, got {type(dtype)}"
    return _NUMPY_TO_TORCH_DTYPE[dtype]
