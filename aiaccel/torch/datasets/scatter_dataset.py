from typing import Callable

import numpy as np
from numpy.random import Generator, PCG64

import torch
import torch.distributed as dist


def scatter_dataset(
    dataset: torch.utils.data.Dataset,
    permute_fn: Callable[[np.ndarray], np.ndarray] = Generator(PCG64(0)).permutation,
):
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    dataset_size = len(dataset)
    total_size = int(np.ceil(dataset_size / world_size)) * world_size

    indices = permute_fn(np.arange(dataset_size))
    repeated_indices = np.concatenate([indices, indices[: total_size - dataset_size]])

    split_indices = np.split(repeated_indices, world_size)

    return torch.utils.data.Subset(dataset, split_indices[rank])
