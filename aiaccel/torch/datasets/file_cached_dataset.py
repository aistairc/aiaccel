from typing import Any, TypeVar

from multiprocessing import Manager
from pathlib import Path
import pickle as pkl
import uuid

from torch.utils.data import Dataset

__all__ = ["FileCachedDataset"]


T_co = TypeVar("T_co", covariant=True)


class FileCachedDataset(Dataset[T_co]):
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
