from collections.abc import Callable
from typing import TypeVar

import numpy as np
import numpy.typing as npt
import torch.distributed as dist
from torch.utils.data import Dataset, Subset

T = TypeVar("T")


def scatter_dataset(
    dataset: Dataset[T],
    permute_fn: Callable[[npt.NDArray[np.int64]], npt.NDArray[np.int64]] | None = None,
) -> Subset[T]:
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
