from pathlib import Path

import pytest
from omegaconf import OmegaConf

import aiaccel
from aiaccel.nas.data_module.get_datasets_api import get_datasets_by_config, get_datasets_by_name, get_project_root


def test_get_project_root():
    project_root = get_project_root()
    assert isinstance(project_root, Path)
    assert project_root.name == "nas"


def test_get_datasets_by_name(mocker):
    mocker.patch("aiaccel.nas.data_module.get_datasets_api.CIFAR10", return_value="CIFAR10 dataset")
    mocker.patch("aiaccel.nas.data_module.get_datasets_api.CIFAR100", return_value="CIFAR100 dataset")
    mocker.patch("aiaccel.nas.data_module.get_datasets_api.MNIST", return_value="MNIST dataset")

    dataset = get_datasets_by_name(name="CIFAR10")
    assert dataset == "CIFAR10 dataset"

    dataset = get_datasets_by_name(name="CIFAR100")
    assert dataset == "CIFAR100 dataset"

    dataset = get_datasets_by_name(name="MNIST")
    assert dataset == "MNIST dataset"

    with pytest.raises(NotImplementedError):
        get_datasets_by_name(name="Unsupported")


def test_get_datasets_by_config(mocker):
    mocker.patch("aiaccel.nas.data_module.get_datasets_api.get_datasets_by_name", return_value="Dataset")

    config = OmegaConf.create(
        {
            "aiaccel": {
                "nas": {
                    "dataloader": {
                        "dataset_name": "CIFAR10",
                        "root": "./data",
                        "train": True,
                        "download": True,
                    },
                },
            },
        },
    )

    dataset = get_datasets_by_config(config=config)
    assert dataset == "Dataset"

    aiaccel.nas.data_module.get_datasets_api.get_datasets_by_name.assert_called_once_with(
        name="CIFAR10",
        root="./data",
        train=True,
        download=True,
    )
