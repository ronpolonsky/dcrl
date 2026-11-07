from typing import Dict, Optional, Union

import numpy as np

from expo_ft.types import DataType

DatasetDict = Dict[str, DataType]


def _check_lengths(dataset_dict: DatasetDict, dataset_len: Optional[int] = None) -> int:
    for v in dataset_dict.values():
        if isinstance(v, dict):
            dataset_len = dataset_len or _check_lengths(v, dataset_len)
        elif isinstance(v, np.ndarray):
            item_len = len(v)
            dataset_len = dataset_len or item_len
            assert dataset_len == item_len, "Inconsistent item lengths in the dataset."
        else:
            raise TypeError("Unsupported type.")
    return dataset_len


class Dataset(object):
    def __init__(self, dataset_dict: DatasetDict, seed: Optional[int] = None):
        self.dataset_dict = dataset_dict
        self.dataset_len = _check_lengths(dataset_dict)

        self._np_random = None
        self._seed = None
        if seed is not None:
            self.seed(seed)

    def seed(self, seed: Optional[int] = None) -> list:
        self._seed = seed
        self._np_random = np.random.RandomState(seed)
        return [self._seed]

    def __len__(self) -> int:
        return self.dataset_len

    def sample(self, batch_size, **kwargs):
        raise NotImplementedError("Random batch sampling from dataset dict.")

    def sample_jax(self, batch_size, **kwargs):
        raise NotImplementedError("JIT-compiled random batch sampling on device.")

    def split(self, ratio):
        raise NotImplementedError("Train/test split by ratio with shuffling.")

    def filter(self, **kwargs):
        raise NotImplementedError("Filter episodes by return threshold or percentile.")

