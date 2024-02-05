from pathlib import Path

import lightning
import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.nn import Module
from torch.optim import SGD

import aiaccel
from aiaccel.nas.asng.categorical_asng import CategoricalASNG
from aiaccel.nas.mnas_structure_info import MnasNetStructureInfo
from aiaccel.nas.module.nas_module import NASModule
from aiaccel.nas.nas_model.mnasnet_lightning_model import (
    MnasnetSearchModel,
    MnasnetTrainModel,
    _create_batch_dependent_lr_scheduler,
    _create_optimizer,
)
from aiaccel.nas.utils.utils import create_config_by_yaml


@pytest.fixture(scope="module")
def parameter_config():
    return OmegaConf.create(
        {
            "batch_size_supernet_train": 256,
            "batch_size_architecture_search": 256,
            "optimizer": "MomentumSGD",
            "initial_lr": 0.001,
            "weight_decay": 0.0,
            "momentum": 0.0,
            "dampening": 0.0,
            "smoothing": 0.99,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1.0e-08,
            "momentum_decay": 0.004,
            "scheduler": "Linear",
            "warmup_epochs": 1,
            "milestone_step": 30,
            "milestone_start": 30,
            "gamma": 0.1,
            "start_factor": 0.3333333333333333,
            "end_factor": 1.0,
            "total_iters": 5,
            "alpha": 1.5,
            "epsilon": 0.0,
            "lam": 8,
            "total_epochs": 2,
        },
    )


def test_create_batch_dependent_lr_scheduler(mocker, parameter_config):
    mocker.patch("aiaccel.nas.nas_model.mnasnet_lightning_model.create_batch_dependent_lr_scheduler", return_value=None)

    optimizer = SGD([{"params": torch.Tensor([1, 2, 3])}], lr=0.01)

    _create_batch_dependent_lr_scheduler(optimizer, parameter_config, num_epochs=10, num_batches=100)

    aiaccel.nas.nas_model.mnasnet_lightning_model.create_batch_dependent_lr_scheduler.assert_called_once_with(
        optimizer,
        {
            "batch_size_supernet_train": 256,
            "batch_size_architecture_search": 256,
            "optimizer": "MomentumSGD",
            "initial_lr": 0.001,
            "weight_decay": 0.0,
            "momentum": 0.0,
            "dampening": 0.0,
            "smoothing": 0.99,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-08,
            "momentum_decay": 0.004,
            "scheduler": "Linear",
            "warmup_epochs": 1,
            "milestone_step": 30,
            "milestone_start": 30,
            "gamma": 0.1,
            "start_factor": 0.3333333333333333,
            "end_factor": 1.0,
            "total_iters": 5,
            "alpha": 1.5,
            "epsilon": 0.0,
            "lam": 8,
            "total_epochs": 2,
        },
        10,
        100,
    )


def test_create_optimizer(mocker, parameter_config):
    mocker.patch("aiaccel.nas.nas_model.mnasnet_lightning_model.create_optimizer", return_value=None)

    nn_module = Module()

    _create_optimizer(nn_module, parameter_config)

    aiaccel.nas.nas_model.mnasnet_lightning_model.create_optimizer.assert_called_once_with(
        nn_module,
        {
            "batch_size_supernet_train": 256,
            "batch_size_architecture_search": 256,
            "optimizer": "MomentumSGD",
            "initial_lr": 0.001,
            "weight_decay": 0.0,
            "momentum": 0.0,
            "dampening": 0.0,
            "smoothing": 0.99,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-08,
            "momentum_decay": 0.004,
            "scheduler": "Linear",
            "warmup_epochs": 1,
            "milestone_step": 30,
            "milestone_start": 30,
            "gamma": 0.1,
            "start_factor": 0.3333333333333333,
            "end_factor": 1.0,
            "total_iters": 5,
            "alpha": 1.5,
            "epsilon": 0.0,
            "lam": 8,
            "total_epochs": 2,
        },
    )


def test_mnasnet_train_model(mocker):
    dataloader = mocker.Mock()
    dataloader.get_dims.return_value = [3, 32, 32]
    dataloader.get_num_classes.return_value = 10
    dataloader.get_num_supernet_train_data.return_value = 100
    # Mock the __iter__ method to return a batch of data
    inputs = torch.randn(32, 3, 32, 32)  # Replace with your actual input tensor shape
    targets = torch.randint(0, 10, (32,))  # Replace with your actual target tensor shape
    batch = (inputs, targets)
    dataloader.__iter__ = mocker.Mock(return_value=iter([batch]))
    for batch_idx, batch in enumerate(dataloader):
        assert batch_idx == 0
        assert batch[0].shape == inputs.shape
        assert batch[1].shape == targets.shape

    # Mock the NASModule
    nn_model = mocker.Mock(spec=NASModule)
    nn_model.parameters = mocker.Mock(
        return_value=[
            torch.Tensor([0.1]),
            torch.Tensor([0.1]),
            torch.Tensor([0.1]),
        ],
    )
    nn_model.forward = mocker.Mock(
        return_value=torch.randn(1, 1, requires_grad=True),
    )
    # nn_model = MnasNetSearchSpace("MnasNetSearchSpace")

    # Mock the CrossEntropyLoss
    loss = nn.CrossEntropyLoss()
    los_func = mocker.Mock(
        spec=nn.CrossEntropyLoss,
        return_value=loss(torch.randn(1, 1, requires_grad=True), torch.empty(1, dtype=torch.long).random_(1)),
    )
    # los_func = nn.CrossEntropyLoss()

    # Create a MnasnetTrainModel instance
    parameter_config = OmegaConf.create(
        {
            "batch_size_supernet_train": 256,
            "batch_size_architecture_search": 256,
            "optimizer": "MomentumSGD",
            "initial_lr": 0.001,
            "weight_decay": 0.0,
            "momentum": 0.0,
            "dampening": 0.0,
            "smoothing": 0.99,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1.0e-08,
            "momentum_decay": 0.004,
            "scheduler": "Linear",
            "warmup_epochs": 1,
            "milestone_step": 30,
            "milestone_start": 30,
            "gamma": 0.1,
            "start_factor": 0.3333333333333333,
            "end_factor": 1.0,
            "total_iters": 5,
            "alpha": 1.5,
            "epsilon": 0.0,
            "lam": 8,
            "total_epochs": 2,
        },
    )
    categories = np.array([1, 2, 3, 4])
    asng = CategoricalASNG(
        categories=categories,
        params=np.array([1, 2, 3, 4]),
    )
    config_path = "./examples/nas/config_proxyless_cifar10.yaml"
    search_space_config = create_config_by_yaml(config_path)
    structure_info = MnasNetStructureInfo([])
    model = MnasnetTrainModel(
        dataloader=dataloader,
        nn_model=nn_model,
        search_space_config=search_space_config,
        parameter_config=parameter_config,
        categories=categories,
        asng=asng,
        structure_info=structure_info,
        los_func=los_func,
        log_dir=Path(),
        parent_logger_name="test",
        nas_config=DictConfig(
            {
                "environment": {"gpus": None},
                "nas": {
                    "search_space": "proxyless",
                    "num_epochs_supernet_train": 2,
                    "num_epochs_architecture_search": 2,
                    "seed": 42,
                },
            },
        ),
        num_train_data_sampler=100,
    )

    # Mock the configure_optimizers and training_step methods
    # mocker.patch.object(MnasnetTrainModel, "configure_optimizers", return_value=None)
    # mocker.patch.object(MnasnetTrainModel, "training_step", return_value=None)

    # Test the forward method
    x = torch.rand(1, 3, 32, 32)
    y = model.forward(x)
    assert y is not None

    # Test the configure_optimizers method
    optimizer = model.configure_optimizers()
    mocker.patch.object(model, "optimizers", return_value=optimizer)
    print("optimizer:", optimizer)
    # nn_model.optimizers = mocker.Mock(return_value=optimizer)

    model._trainer = lightning.Trainer()
    model.on_train_start()
    assert model.start is not None

    # Test the training_step method
    batch = (torch.rand(10, 3, 32, 32), torch.randint(0, 10, (10,)))
    batch_idx = 0
    for batch_idx, batch in enumerate(dataloader):
        model.training_step(batch, batch_idx)

    model.on_train_epoch_start()
    assert model.train_loss == 0.0
    assert model.train_acc1 == 0.0
    assert model.train_acc5 == 0.0

    mocker.patch.object(model, "log", return_value=None)
    model.on_train_epoch_end()
    assert model.custom_logger is not None


def test_mnasnet_search_model(mocker):
    # Mock the DataLoader
    dataloader = mocker.Mock()
    dataloader.get_dims.return_value = [3, 32, 32]
    dataloader.get_num_classes.return_value = 10
    dataloader.get_num_architecture_search_data.return_value = 100

    # Mock the NASModule
    nn_model = mocker.Mock(spec=NASModule)

    # Mock the CrossEntropyLoss
    los_func = mocker.Mock(spec=nn.CrossEntropyLoss)

    # Create a MnasnetSearchModel instance
    model = MnasnetSearchModel(
        dataloader=dataloader,
        nn_model=nn_model,
        search_space_config=(1, None),
        parameter_config=DictConfig({"lam": 1}),
        categories=None,
        asng=None,
        structure_info=None,
        los_func=los_func,
        log_dir=Path(),
        parent_logger_name="test",
        nas_config=DictConfig({"environment": {"gpus": None}}),
        num_train_data_sampler=100,
    )

    # Mock the configure_optimizers and training_step methods
    mocker.patch.object(MnasnetSearchModel, "configure_optimizers", return_value=None)
    mocker.patch.object(MnasnetSearchModel, "training_step", return_value=None)

    # Test the forward method
    x = torch.rand(1, 3, 32, 32)
    y = model.forward(x)
    assert y is not None

    # Test the configure_optimizers method
    model.configure_optimizers()

    # Test the training_step method
    batch = (torch.rand(10, 3, 32, 32), torch.randint(0, 10, (10,)))
    batch_idx = 0
    model.training_step(batch, batch_idx)

    model.on_train_start()
    assert model.start is not None

    model.on_train_epoch_start()
    assert model.valid_loss == 0.0
    assert model.valid_acc1 == 0.0
    assert model.valid_acc5 == 0.0

    model.asng = CategoricalASNG(
        categories=np.array([1, 2, 3, 4]),
        params=np.array([1, 2, 3, 4]),
    )
    model.on_train_epoch_end()
    assert model.custom_logger is not None
