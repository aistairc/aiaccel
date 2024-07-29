from __future__ import annotations

from typing import Any

from pathlib import Path
import pickle as pkl

import torch
from torch import distributed as dist
from torch.utils.data import Dataset

import h5py as h5

__all__ = [
    "RawHDF5Dataset",
    "HDF5Dataset",
]


class RawHDF5Dataset(Dataset[Any]):
    def __init__(self, dataset_path: Path | str, grp_list: Path | str | list[str] | None = None) -> None:
        self.dataset_path = dataset_path

        if not dist.is_initialized() or dist.get_rank() == 0:
            if grp_list is None:
                with h5.File(self.dataset_path, "r") as f:
                    self.grp_list = list(f.keys())
            elif isinstance(grp_list, str | Path):
                with open(grp_list, "rb") as f:
                    self.grp_list = pkl.load(f)
            elif isinstance(grp_list, list):
                self.grp_list = grp_list
            else:
                raise NotImplementedError()
            self.grp_list.sort()

        if dist.is_initialized():
            bc_obj_list = [self.grp_list] if dist.get_rank() == 0 else [None]  # type: ignore
            dist.broadcast_object_list(bc_obj_list, src=0)

            self.grp_list = bc_obj_list[0]

    def __len__(self) -> int:
        return len(self.grp_list)

    def __getitem__(self, index: int) -> dict[str, Any]:
        with h5.File(self.dataset_path, "r") as f:
            ret = {k: v[:] for k, v in f[self.grp_list[index]].items()}

        return ret


class HDF5Dataset(RawHDF5Dataset):
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {k: torch.as_tensor(v) for k, v in super().__getitem__(index).items()}
