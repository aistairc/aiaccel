from __future__ import annotations

from typing import Any

from pathlib import Path
import pickle as pkl

import torch
from torch.utils.data import Dataset

import h5py as h5

__all__ = [
    "RawHDF5Dataset",
    "HDF5Dataset",
]


class RawHDF5Dataset(Dataset[dict[str, Any]]):
    """
    A dataset class for reading data from HDF5 files.

    Args:
        dataset_path (Union[Path, str]): The path to the HDF5 dataset file.
        grp_list (Union[Path, str, List[str], None], optional): The list of groups to load from the dataset.
            If None, all groups in the dataset will be loaded. If a string or Path, it should be the path to a file
            containing the list of groups. If a list, it should directly specify the groups to load. Defaults to None.

    Raises:
        NotImplementedError: If grp_list is of an unsupported type.

    Attributes:
        dataset_path (Union[Path, str]): The path to the HDF5 dataset file.
        grp_list (List[str]): The list of groups to load from the dataset.
        f (Optional[h5.File]): The HDF5 file object used for reading the dataset.

    """

    def __init__(self, dataset_path: Path | str, grp_list: Path | str | list[str] | None = None) -> None:
        self.dataset_path = dataset_path

        if grp_list is None:
            with h5.File(self.dataset_path, "r") as f:
                self.grp_list = list(f.keys())
        elif isinstance(grp_list, (str | Path)):
            with open(grp_list, "rb") as f:
                self.grp_list = pkl.load(f)
        elif isinstance(grp_list, list):
            self.grp_list = grp_list
        else:
            raise NotImplementedError()
        self.grp_list.sort()

        self.f: h5.File | None = None

    def __len__(self) -> int:
        return len(self.grp_list)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.f is None:
            self.f = h5.File(self.dataset_path, "r")

        return {k: v[:] for k, v in self.f[self.grp_list[index]].items()}

    def __del__(self) -> None:
        if self.f is not None:
            self.f.close()


class HDF5Dataset(RawHDF5Dataset):
    """
    A dataset class for loading data from an HDF5 file.

    This class extends the `RawHDF5Dataset` class and provides a convenient way to load data from an HDF5 file
    and convert it into a dictionary of torch tensors.

    Args:
        path (str): The path to the HDF5 file.
        transform (callable, optional): A function/transform that takes in a dictionary of data and returns a
            modified version. Default is None.

    Returns:
        dict[str, torch.Tensor]: A dictionary containing the data loaded from the HDF5 file, where the keys are
            the names of the data fields and the values are torch tensors.
    """

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {k: torch.as_tensor(v) for k, v in super().__getitem__(index).items()}
