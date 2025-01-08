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
        super().__init__()

        self.train_dataset_fn = train_dataset_fn
        self.val_dataset_fn = val_dataset_fn

        self.default_dataloader_kwargs = dict[str, Any](
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=True,
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
            **self.default_dataloader_kwargs,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return DataLoader(
            self.val_dataset,
            drop_last=False,
            **self.default_dataloader_kwargs,
        )
