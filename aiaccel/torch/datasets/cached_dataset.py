# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing import Any, TypeVar

from multiprocessing import Manager

import torch
from torch.utils.data import Dataset

__all__ = ["CachedDataset"]


class NumpiedTensor:
    """
    A wrapper class that converts a PyTorch tensor to a NumPy array and vice versa.

    Args:
        tensor (torch.Tensor): The input PyTorch tensor.

    Attributes:
        array (np.ndarray): The NumPy array representation of the tensor.

    Methods:
        to_tensor: Converts the NumPy array back to a PyTorch tensor.
    """

    def __init__(self, tensor: torch.Tensor) -> None:
        self.array = tensor.numpy()

    def to_tensor(self) -> torch.Tensor:
        """
        Converts the NumPy array back to a PyTorch tensor.

        Returns:
            torch.Tensor: The PyTorch tensor representation of the NumPy array.
        """
        return torch.tensor(self.array)


def numpize_sample(sample: Any) -> Any:
    """
    Converts the input sample to a NumPy-compatible format.

    Args:
        sample (Any): The input sample to be converted.

    Returns:
        Any: The converted sample in a NumPy-compatible format.
    """

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
    """
    Converts the given sample into a tensor representation.

    Args:
        sample (Any): The input sample to be tensorized.

    Returns:
        Any: The tensorized representation of the input sample.
    """

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


T_co = TypeVar("T_co", covariant=True)


class CachedDataset(Dataset[T_co]):
    """
    A dataset wrapper that caches the samples to improve performance.

    Args:
        dataset (Dataset): The original dataset to be wrapped.

    Attributes:
        dataset (Dataset): The original dataset.
        manager (Manager): The multiprocessing manager.
        cache (dict): The cache dictionary to store the cached samples.
    """

    def __init__(self, dataset: Dataset[T_co]) -> None:
        self.dataset = dataset

        self.manager = Manager()
        self.cache = self.manager.dict()

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> Any:
        if index not in self.cache:
            self.cache[index] = numpize_sample(self.dataset[index])

        return tensorize_sample(self.cache[index])
