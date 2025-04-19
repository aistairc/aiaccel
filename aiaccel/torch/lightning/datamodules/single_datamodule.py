from typing import Any

from collections.abc import Callable, Sized

from torch.utils.data import DataLoader, Dataset, Subset

import lightning as lt

from aiaccel.torch.datasets import scatter_dataset


class SingleDataModule(lt.LightningDataModule):
    def __init__(
        self,
        train_dataset_fn: Callable[..., Dataset[str]],
        val_dataset_fn: Callable[..., Dataset[str]],
        batch_size: int,
        num_workers: int = 10,
        wrap_scatter_dataset: bool = True,
    ):
        """
        PyTorch Lightning DataModule for a single training/validation dataset pair.

        This module initializes training and validation datasets using the provided
        dataset functions and wraps them using `scatter_dataset` if specified.
        It sets up corresponding dataloaders with configurable batch size, worker count,
        and shuffling behavior.

        Args:
            train_dataset_fn (Callable[..., Dataset[str]]):
                A function that returns the training dataset.
            val_dataset_fn (Callable[..., Dataset[str]]):
                A function that returns the validation dataset.
            batch_size (int):
                Batch size for the dataloaders.
            num_workers (int, optional):
                Number of workers for data loading. Defaults to 10.
            wrap_scatter_dataset (bool, optional):
                Whether to wrap datasets using `scatter_dataset`. Defaults to True.

        Attributes:
            train_dataset (Dataset[str] | Subset[str]):
                Initialized training dataset after optional wrapping.
            val_dataset (Dataset[str] | Subset[str]):
                Initialized validation dataset after optional wrapping.
        """

        super().__init__()

        self.train_dataset_fn = train_dataset_fn
        self.val_dataset_fn = val_dataset_fn

        self._default_dataloader_kwargs = dict[str, Any](
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=True,
            pin_memory=True,
            shuffle=True,
        )

        self.wrap_scatter_dataset = wrap_scatter_dataset

    def setup(self, stage: str | None) -> None:
        self.train_dataset: Dataset[str] | Subset[str]
        self.val_dataset: Dataset[str] | Subset[str]
        if stage == "fit":
            if self.wrap_scatter_dataset:
                self.train_dataset = scatter_dataset(self.train_dataset_fn())
                self.val_dataset = scatter_dataset(self.val_dataset_fn())
            else:
                self.train_dataset = self.train_dataset_fn()
                self.val_dataset = self.val_dataset_fn()

            if isinstance(self.train_dataset, Sized) and isinstance(self.val_dataset, Sized):
                print(f"Dataset size: {len(self.train_dataset)=},  {len(self.val_dataset)=}")
        else:
            raise ValueError("`stage` is not 'fit'.")

    def train_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.train_dataset,
            drop_last=True,
            **self._default_dataloader_kwargs,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.val_dataset,
            drop_last=False,
            **self._default_dataloader_kwargs,
        )
