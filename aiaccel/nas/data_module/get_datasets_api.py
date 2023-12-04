from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from nas.data_module.cifar10_data_module import CIFAR10DataModule
from torchvision.datasets import CIFAR10, CIFAR100, MNIST

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch.utils.data import Dataset


def get_project_root() -> Path:
    """
    Returns the root path of the project.
    """
    return Path(__file__).parent.parent


def get_datasets_by_name(
    name: str | None = None,
    root: str = "./data",
    *,
    train: bool = True,
    download: bool = True,
) -> Dataset:
    """Generate a dataset by name.

    Args:
        name (str, optional): A dataset name. Defaults to None.
        root (str, optional): A root directory to place the dataset. Defaults to "./data".
        train (bool, optional): An option whether load training data or not. Defaults to True.
        download (bool, optional): An option whether download or not. Defaults to True.

    Raises:
        NotImplementedError: No matched dataset name.

    Returns:
        Dataset: A dataset object.
    """
    if name == "CIFAR10":
        return CIFAR10(root=root, train=train, download=download)

    if name == "CIFAR100":
        return CIFAR100(root=root, train=train, download=download)

    if name == "MNIST":
        return MNIST(root=root, train=train, download=download)

    raise NotImplementedError


def get_datasets_by_config(config: DictConfig) -> Dataset:
    """Generate a dataset by config.

    Args:
        config (DictConfig): OmegaConf object.

    Returns:
        Dataset: A dataset object.
    """
    name = config.aiaccel.nas.dataloader.dataset_name
    root = config.aiaccel.nas.dataloader.root
    train = config.aiaccel.nas.dataloader.train
    download = config.aiaccel.nas.dataloader.download

    return get_datasets_by_name(name=name, root=root, train=train, download=download)
