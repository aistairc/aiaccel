from typing import Any, Callable

import numpy as np
import torch
import torch.distributed as dist
from numpy.random import PCG64, Generator


def scatter_dataset(
    dataset: torch.utils.data.Dataset,
    permute_fn: Callable[[np.ndarray[Any, np.dtype[np.int64]]], np.ndarray[Any, np.dtype[np.int64]]] = Generator(PCG64(0)).permutation,
) -> torch.utils.data.Subset:
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    dataset_size = len(dataset)
    total_size = int(np.ceil(dataset_size / world_size)) * world_size

    indices = permute_fn(np.arange(dataset_size))
    repeated_indices = np.concatenate([indices, indices[: total_size - dataset_size]])

    split_indices = np.split(repeated_indices, world_size)

    return torch.utils.data.Subset(dataset, split_indices[rank])
