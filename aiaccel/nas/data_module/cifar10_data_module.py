from __future__ import annotations

from typing import TYPE_CHECKING

import lightning
from nas.data_module.subset_random_sampling_dataloader import SubsetRandomSamplingDataLoader
from nas.utils.utils import _data_transforms_cifar, get_random_indices
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split
from torchvision.datasets import CIFAR10

if TYPE_CHECKING:
    from pathlib import Path


class Cifar10DataModule(lightning.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        train_batch_size: int,
        test_batch_size: int,
        num_workers: int = 1,
        *,
        persistent_workers: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.transform, _ = _data_transforms_cifar()
        self.dims = (3, 32, 32)
        self.num_classes = 10
        self.num_train_data = 0
        self.num_test_data = 0

    def prepare_data(self):
        # download
        train_dataset = CIFAR10(self.data_dir, train=True, download=True)
        test_dataset = CIFAR10(self.data_dir, train=False, download=True)
        self.num_train_data = len(train_dataset)
        self.num_test_data = len(test_dataset)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(self.cifar_full, [45000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.cifar_train,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.cifar_val,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.cifar_test,
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )


class Cifar10SubsetRandomSamplingDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int,
        training_mode: str = "supernet_train",
        num_data_architecture_search: int = 10000,
        num_workers: int = 1,
        *,
        persistent_workers: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.transform, _ = _data_transforms_cifar()
        self.dims = (3, 32, 32)
        self.num_classes = 10
        self.training_mode = training_mode
        self.num_data_architecture_search = num_data_architecture_search
        self.num_train_data = 0
        self.num_test_data = 0
        self.sampler = None

    def prepare_data(self):
        # download
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.cifar_full = CIFAR10(self.data_dir, train=True, transform=self.transform)
            self.cifar_train, self.cifar_val = random_split(self.cifar_full, [45000, 5000])

        if stage == "test" or stage is None:
            self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.transform)
        # self.num_train_data = len(self.cifar_train)
        self.num_train_data = len(self.cifar_full)
        self.num_test_data = len(self.cifar_test)
        print(
            "Cifar10SubsetRandomSamplingDataModule,",
            "self.num_train_data:",
            self.num_train_data,
            "self.num_test_data:",
            self.num_test_data,
            "self.num_data_architecture_search:",
            self.num_data_architecture_search,
        )
        architecture_search_data_indices, supernet_train_data_indices = get_random_indices(
            self.num_train_data,
            self.num_data_architecture_search,
        )
        if self.training_mode == "supernet_train":
            self.sampler = SubsetRandomSampler(supernet_train_data_indices)
        elif self.training_mode == "architecture_search":
            self.sampler = SubsetRandomSampler(architecture_search_data_indices)
        else:
            raise ValueError(self.training_mode)

    def train_dataloader(self):
        return DataLoader(
            self.cifar_full,
            batch_size=self.batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.cifar_val,
            batch_size=self.batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.cifar_test,
            batch_size=self.batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
        )


def get_cifar10_dataset(data_dir: Path, *, is_test: bool = False, is_download: bool = True):
    transform, _ = _data_transforms_cifar()
    CIFAR10(data_dir, train=(not is_test), download=is_download)
    return CIFAR10(data_dir, train=(not is_test), transform=transform)


def get_cifar10_dataloader(
    dataset,
    batch_size: int,
    index: int,
    num_workers: int = 1,
    *,
    persistent_workers: bool = True,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(index),
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )


class Cifar10SubsetRandomSamplingDataLoader(SubsetRandomSamplingDataLoader):
    def __init__(
        self,
        data_dir: Path,
        batch_size_supernet_train: int,
        batch_size_architecture_search: int,
        num_data_architecture_search: int = 10000,
        num_workers: int = 1,
        *,
        persistent_workers: bool = True,
    ):
        self._data_dir = data_dir
        self._batch_size_supernet_train = batch_size_supernet_train
        self._batch_size_architecture_search = batch_size_architecture_search
        self._num_workers = num_workers
        self._persistent_workers = persistent_workers
        self._dims = (3, 32, 32)
        self._num_classes = 10

        CIFAR10(data_dir, train=True, download=True)
        _transform, _ = _data_transforms_cifar()
        self._cifar_full = CIFAR10(self._data_dir, train=True, transform=_transform)
        self._num_train_data = len(self._cifar_full)
        self._architecture_search_data_indices, self._supernet_train_data_indices = get_random_indices(
            self._num_train_data,
            num_data_architecture_search,
        )
        # self._sampler_supernet_train = SubsetRandomSampler(_supernet_train_data_indices)
        # self._sampler_architecture_search = SubsetRandomSampler(_architecture_search_data_indices)
        self._create_sampler()

    def get_supernet_train_dataloader(self) -> DataLoader:
        """Get a data loader for the supernet training.

        Returns:
            DataLoader: A data loader sampled using SubsetRandomSampler. The index is generated by get_random_indices().
        """
        return DataLoader(
            self._cifar_full,
            batch_size=self._batch_size_supernet_train,
            sampler=self._sampler_supernet_train,
            num_workers=self._num_workers,
            persistent_workers=self._persistent_workers,
        )

    def get_architecture_search_dataloader(self) -> DataLoader:
        """Get a data loader for the architecture search.

        Returns:
            DataLoader: A data loader sampled using SubsetRandomSampler. The index is generated by get_random_indices().
        """
        return DataLoader(
            self._cifar_full,
            batch_size=self._batch_size_architecture_search,
            sampler=self._sampler_architecture_search,
            num_workers=self._num_workers,
            persistent_workers=self._persistent_workers,
        )

    def get_num_supernet_train_data(self) -> int:
        """Get a number of supernet train data.

        Returns:
            int: A number of supernet train data.
        """
        return self._num_train_data

    def get_dims(self) -> tuple[int, int, int]:
        """Get dimensions of images of the data loaders.

        Returns:
            tuple[int, int, int]: Dimensions of images of the data loaders.
        """
        return self._dims

    def get_num_classes(self) -> int:
        """Get a number of classes of the data loaders.

        Returns:
            int: A number of classes of the data loaders.
        """
        return self._num_classes
