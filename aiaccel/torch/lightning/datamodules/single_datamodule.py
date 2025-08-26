from typing import Any

from collections.abc import Callable

from torch.utils.data import DataLoader, Dataset

import lightning as lt

from aiaccel.torch.datasets import CachedDataset, scatter_dataset


class SingleDataModule(lt.LightningDataModule):
    """
    A PyTorch Lightning DataModule designed to handle training and validation datasets
    with support for caching and dataset scattering.

    Attributes:
        train_dataset_fn (Callable[..., Dataset[str]]): A callable function to create the training dataset.
        val_dataset_fn (Callable[..., Dataset[str]]): A callable function to create the validation dataset.
        batch_size (int): The batch size for the DataLoader.
        use_cache (bool): Whether to cache the datasets. Defaults to False.
        use_scatter (bool): Whether to scatter the datasets. Defaults to True.
        num_workers (int): Number of workers for the DataLoader. Defaults to 10.
        common_args (dict[str, Any] | None): Common arguments to pass to the dataset functions. Defaults to None.
    Methods:
        setup(stage: str | None) -> None:
            Prepares the datasets for training and validation. Only supports the "fit" stage.
            Raises a ValueError if the stage is not "fit".
        train_dataloader() -> DataLoader:
            Returns the DataLoader for the training dataset.
        val_dataloader() -> DataLoader:
            Returns the DataLoader for the validation dataset.
        _create_dataloader(dataset, **kwargs: Any) -> DataLoader:
            Internal method to create a DataLoader for a given dataset with specified configurations.
    """

    def __init__(
        self,
        train_dataset_fn: Callable[..., Dataset[str]],
        val_dataset_fn: Callable[..., Dataset[str]],
        batch_size: int,
        use_cache: bool = False,
        use_scatter: bool = True,
        num_workers: int = 10,
        common_args: dict[str, Any] | None = None,
    ):
        super().__init__()

        self.train_dataset_fn = train_dataset_fn
        self.val_dataset_fn = val_dataset_fn

        self.common_args = common_args if common_args is not None else {}

        self.batch_size = batch_size

        self.use_cache = use_cache
        self.use_scatter = use_scatter

        self.num_workers = num_workers

    def setup(self, stage: str | None) -> None:
        if stage == "fit":
            train_dataset = self.train_dataset_fn(**self.common_args)
            val_dataset = self.val_dataset_fn(**self.common_args)

            print(f"Dataset size: {len(train_dataset)=},  {len(val_dataset)=}")  # type: ignore

            if self.use_cache:
                train_dataset = CachedDataset(train_dataset)
                val_dataset = CachedDataset(val_dataset)

            if self.use_scatter:
                train_dataset = scatter_dataset(train_dataset)
                val_dataset = scatter_dataset(val_dataset)

            self.train_dataset = train_dataset
            self.val_dataset = val_dataset
        else:
            raise ValueError("`stage` is not 'fit'.")

    def _create_dataloader(self, dataset: Dataset[Any], **kwargs: Any) -> DataLoader[Any]:
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True,
            shuffle=True,
            pin_memory=True,
            **kwargs,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        return self._create_dataloader(self.train_dataset, drop_last=True)

    def val_dataloader(self) -> DataLoader[Any]:
        return self._create_dataloader(self.val_dataset, drop_last=False)
