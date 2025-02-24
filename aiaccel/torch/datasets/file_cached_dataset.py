from typing import Any, TypeVar

from multiprocessing import Manager
from pathlib import Path
import pickle as pkl
import uuid

from torch.utils.data import Dataset

__all__ = ["FileCachedDataset"]


T_co = TypeVar("T_co", covariant=True)


class FileCachedDataset(Dataset[T_co]):
    """
    A dataset wrapper that caches samples to disk to reduce memory usage.

    This class wraps an existing PyTorch Dataset and caches samples as pickle
    files in a specified directory. It is useful when working with large datasets
    that do not fit in memory.

    Attributes:
        dataset (Dataset[T_co]): The original dataset to wrap.
        cache_path (Path): Directory where cached samples are stored.
        manager (Manager): A multiprocessing manager for shared state.
        cache (dict): A dictionary mapping sample indices to cache file paths.

    Args:
        dataset (Dataset[T_co]): The dataset to wrap.
        cache_path (str | Path): Directory where cached samples will be stored.

    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(index: int) -> Any: Retrieves a sample from cache or the original dataset.
    """

    def __init__(self, dataset: Dataset[T_co], cache_path: str | Path) -> None:
        self.dataset = dataset

        self.manager = Manager()
        self.cache = self.manager.dict()

        self.cache_path = Path(cache_path)
        self.cache_path.mkdir(exist_ok=True, parents=True)

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> Any:
        if index not in self.cache:
            sample = self.dataset[index]

            self.cache[index] = self.cache_path / f"cache-{uuid.uuid4()}.pkl"
            with open(self.cache[index], "wb") as f:
                pkl.dump(sample, f)
        else:
            with open(self.cache[index], "rb") as f:
                sample = pkl.load(f)

        return sample
