# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import numpy.typing as npt
from typing import Any, Generic, TypeVar

from abc import ABCMeta, abstractmethod
from functools import reduce
import json
from math import ceil
from pathlib import Path

from rich.progress import track

import h5py

T1 = TypeVar("T1")
T2 = TypeVar("T2")


class HDF5Writer(Generic[T1, T2], metaclass=ABCMeta):
    """
    Abstract base class for writing data to an HDF5 file.

    This class provides methods to write data into HDF5 format, supporting both
    single-process and parallel (MPI-based) writing. Subclasses must implement
    `prepare_globals` and `prepare_group` to define how data is structured.

    Typical usage is supposed to be:

    .. code-block:: python

        class FooHDF5Writer(HDF5Writer):
            def prepare_globals(self):
                item_list = list(range(100))

                offset = 10
                maximum = 50

                return item_list, (offset, maximum)

            def prepare_group(self, item, context):
                offset, maximum = context

                group_name = f"{item:04d}

                return {group_name: {"data": np.full([10, 10], offset + item).clip(maximum)}}

        writer = FooHDF5Writer()
        writer.write("test.hdf5", parallel=False)
    """

    h5: h5py.File

    def _write(self, filename: Path) -> None:
        """
        Write data to an HDF5 file using a single process.

        Args:
            filename (Path): Path to the output HDF5 file.
        """

        # prepare globals
        items, context = self.prepare_globals()
        group_list = []

        # write into hdf5 file
        with h5py.File(filename, "w") as h5:
            for item in track(items):
                groups = self.prepare_group(item, context)

                for group_name, datasets in groups.items():
                    g = h5.create_group(group_name)

                    for dataset_name, data in datasets.items():
                        ds = g.create_dataset(dataset_name, data.shape, dtype=data.dtype)
                        ds[:] = data

                    group_list.append(group_name)

        with open(filename.with_suffix(".json"), "w") as f:
            json.dump(group_list, f)

    def _write_parallel(self, filename: Path) -> None:
        """
        Write data to an HDF5 file using MPI for parallel processing.

        Args:
            filename (Path): Path to the output HDF5 file.
        """

        # prepare MPI
        from mpi4py.MPI import COMM_WORLD

        comm = COMM_WORLD

        rank = comm.Get_rank()
        size = comm.Get_size()

        # prepare globals
        if rank == 0:
            items, context = self.prepare_globals()
            items = list(items) + (ceil(len(items) / size) * size - len(items)) * [None]

            globals_ = items, context
        else:
            globals_ = None

        items, context = comm.bcast(globals_, root=0)
        group_list = []

        # write into hdf5 file
        with h5py.File(filename, "w", driver="mpio", comm=comm) as h5:
            track_ = track if rank == 0 else lambda x, **kwargs: x
            for item in track_(items[rank::size]):
                groups = self.prepare_group(item, context) if item is not None else {}

                groups_info = {}
                for group_name, datasets in groups.items():
                    groups_info[group_name] = {dset: (data.shape, data.dtype) for dset, data in datasets.items()}

                for group_name, datasets in reduce(dict.__or__, comm.allgather(groups_info)).items():
                    g = h5.create_group(group_name)

                    for dataset_name, (shape, dtype) in datasets.items():
                        g.create_dataset(dataset_name, shape, dtype=dtype)

                    group_list.append(group_name)

                for group_name, datasets in groups.items():
                    g = h5[group_name]  # type: ignore

                    for dataset_name, data in datasets.items():
                        g[dataset_name][:] = data  # type: ignore

        if rank == 0:
            with open(filename.with_suffix(".json"), "w") as f:
                json.dump(group_list, f)

    def write(self, filename: Path, parallel: bool = False) -> None:
        """
        Write data to an HDF5 file, optionally using parallel processing.

        Args:
            filename (Path): Path to the output HDF5 file.
            parallel (bool, optional): Whether to use parallel writing. Defaults to False.
        """

        if not parallel:
            self._write(filename)
        else:
            self._write_parallel(filename)

    @abstractmethod
    def prepare_globals(self) -> tuple[list[T1], T2]:
        """
        Prepare the global data required for writing.

        This method must be implemented by subclasses to provide the data items
        and any necessary context for processing.

        Returns:
            tuple[list[T1], T2]: A tuple containing a list of data items and
            context information.
        """
        pass

    @abstractmethod
    def prepare_group(self, item: T1, context: T2) -> dict[str, dict[str, npt.NDArray[Any]]]:
        """
        Prepare groups of datasets for writing to HDF5.

        This method must be implemented by subclasses to define how individual
        data items should be structured within the HDF5 file.

        Args:
            item (T1): A single data item.
            context (T2): Additional context for processing.

        Returns:
            dict[str, dict[str, npt.NDArray[Any]]]: A dictionary mapping group names
            to dataset dictionaries.
        """
        pass
