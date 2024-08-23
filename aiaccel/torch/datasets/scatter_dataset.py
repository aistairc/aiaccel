import numpy.typing as npt
from typing import TypeVar

from collections.abc import Callable

import numpy as np

import torch.distributed as dist
from torch.utils.data import Dataset, Subset

T = TypeVar("T")


def scatter_dataset(
    dataset: Dataset[T],
    permute_fn: Callable[[npt.NDArray[np.int64]], npt.NDArray[np.int64]] | None = None,
) -> Subset[T]:
    """
    Splits a dataset into subsets and returns the subset corresponding to the current process rank.

    Args:
        dataset (Dataset[T]): The input dataset to be split.
        permute_fn (Callable[[npt.NDArray[np.int64]], npt.NDArray[np.int64]] | None, optional):
            A function that takes an array of indices and returns a permuted version of the array.
            If None, a default permutation function using np.random.Generator is used.
            Defaults to None.

    Returns:
        Subset[T]: The subset of the input dataset corresponding to the current process rank.
    """

    if permute_fn is None:
        permute_fn = np.random.Generator(np.random.PCG64(0)).permutation

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    dataset_size = len(dataset)  # type: ignore[arg-type]
    total_size = int(np.ceil(dataset_size / world_size)) * world_size

    indices = permute_fn(np.arange(dataset_size))
    repeated_indices = np.concatenate([indices, indices[: total_size - dataset_size]])

    split_indices = np.split(repeated_indices, world_size)

    return Subset(dataset, list(split_indices[rank]))
