from pathlib import Path

import numpy as np

import torch

import h5py as h5

from aiaccel.torch.datasets.hdf5_dataset import HDF5Dataset, RawHDF5Dataset

# with h5.File(Path(__file__).parent / "test_hdf5_dataset_assets" / "dataset.hdf5", "w") as f:
#     for ii in range(10):
#         g = f.create_group(f"grp{ii}")  # noqa: ERA001
#         g.create_dataset("foo", [2, 3, 4])  # noqa: ERA001
#         g.create_dataset("bar", [5, 6])  # noqa: ERA001

#         g["foo"][:] = np.random.randn(2, 3, 4)  # noqa: ERA001
#         g["bar"][:] = np.random.randn(5, 6)  # noqa: ERA001


def test_raw_hdf5_dataset() -> None:
    hdf5_filename = Path(__file__).parent / "test_hdf5_dataset_assets" / "dataset.hdf5"
    f_hdf5 = h5.File(hdf5_filename)

    dataset = RawHDF5Dataset(hdf5_filename)

    assert len(dataset) == 10
    assert list(dataset.grp_list) == [f"grp{idx}" for idx in range(10)]

    sample = dataset[5]
    assert sorted(sample.keys()) == ["bar", "foo"]
    assert np.array_equal(sample["bar"], f_hdf5["grp5"]["bar"][:])
    assert np.array_equal(sample["foo"], f_hdf5["grp5"]["foo"][:])


def test_hdf5_dataset() -> None:
    hdf5_filename = Path(__file__).parent / "test_hdf5_dataset_assets" / "dataset.hdf5"
    f_hdf5 = h5.File(hdf5_filename)

    dataset = HDF5Dataset(hdf5_filename)

    assert len(dataset) == 10
    assert list(dataset.grp_list) == [f"grp{idx}" for idx in range(10)]

    sample = dataset[5]
    assert sorted(sample.keys()) == ["bar", "foo"]
    assert isinstance(sample["bar"], torch.Tensor)
    assert isinstance(sample["foo"], torch.Tensor)
    assert np.array_equal(sample["bar"].numpy(), f_hdf5["grp5"]["bar"][:])
    assert np.array_equal(sample["foo"].numpy(), f_hdf5["grp5"]["foo"][:])
