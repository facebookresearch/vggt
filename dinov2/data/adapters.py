# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Any, Tuple

from torch.utils.data import Dataset


class DatasetWithEnumeratedTargets(Dataset):
    def __init__(self, dataset):
        self._dataset = dataset

    def get_image_data(self, index: int) -> bytes:
        return self._dataset.get_image_data(index)

    def get_target(self, index: int) -> Tuple[Any, int]:
        target = self._dataset.get_target(index)
        return (index, target)

    def __getitem__(self, index: int) -> Tuple[Any, Tuple[Any, int]]:
        image, target = self._dataset[index]
        target = index if target is None else target
        return image, (index, target)

    def __len__(self) -> int:
        return len(self._dataset)
