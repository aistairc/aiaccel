from typing import Any

from multiprocessing import Manager

import numpy as np

import torch


class NumpiedTensor:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.array = tensor.numpy()

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(self.array)


def numpize_sample(sample: Any) -> Any:
    if isinstance(sample, torch.Tensor):
        return NumpiedTensor(sample)
    elif isinstance(sample, tuple):
        return tuple(numpize_sample(s) for s in sample)
    elif isinstance(sample, list):
        return [numpize_sample(s) for s in sample]
    elif isinstance(sample, dict):
        return {k: numpize_sample(v) for k, v in sample.items()}
    else:
        return sample


def tensorize_sample(sample: Any) -> Any:
    if isinstance(sample, NumpiedTensor):
        return sample.to_tensor()
    elif isinstance(sample, tuple):
        return tuple(tensorize_sample(s) for s in sample)
    elif isinstance(sample, list):
        return [tensorize_sample(s) for s in sample]
    elif isinstance(sample, dict):
        return {k: tensorize_sample(v) for k, v in sample.items()}
    else:
        return sample


class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset) -> None:
        self.dataset = dataset

        self.manager = Manager()
        self.cache = self.manager.dict()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        if index not in self.cache:
            self.cache[index] = numpize_sample(self.dataset[index])

        return tensorize_sample(self.cache[index])
