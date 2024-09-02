from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import h5py as h5
import torch


class RawHDF5Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path: Path | str) -> None:
        self.dataset_path = dataset_path

        with h5.File(self.dataset_path, "r") as f:
            self.grp_list = sorted(list(f.keys()))

        self.f: Optional[h5.File] = None

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
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {k: torch.as_tensor(v) for k, v in super().__getitem__(index).items()}
