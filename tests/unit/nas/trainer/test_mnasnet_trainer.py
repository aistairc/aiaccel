from pathlib import Path

import pytest
from omegaconf import DictConfig

from aiaccel.nas.trainer.mnasnet_trainer import MnasnetTrainer


@pytest.fixture()
def mock_mnasnet_trainer():
    nas_config = DictConfig(
        {
            "dataset": {
                "name": "cifar10",
                "num_search_classes": 10,
                "num_data_architecture_search": 10000,
                "path": "./data",
                "retrain": {
                    "train": "./data",
                    "test": "./data",
                },
            },
            "environment": {
                "device_ids": [0],
                "gpus": None,
                "num_workers": 4,
                "result_dir": "/tmp",
            },
            "nas": {
                "num_epochs_architecture_search": 2,
                "num_epochs_supernet_train": 2,
                "num_epochs_retrain": 2,
                "search_space": "proxyless",
                "search_space_config_path": "./examples/nas/",
                "seed": 42,
                "skip_architecture_search": False,
                "skip_retrain": False,
                "skip_train": False,
            },
            "retrain": {
                "hyperparameters": {
                    "base_lr": 0.01,
                    "batch_size": 64,
                    "momentum": 0.0,
                    "num_epochs": 10,
                    "weight_decay": 0.0,
                },
            },
            "trainer": {
                "accelerator": "gpu",
                "enable_model_summary": False,
                "enable_progress_bar": False,
                "logger": False,
                "strategy": "auto",
            },
        },
    )
    parameter_config = DictConfig(
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
    trainer = MnasnetTrainer(nas_config, parameter_config)
    return trainer


def test_train(mock_mnasnet_trainer, mocker):
    trainer_mock = mocker.Mock()
    mocker.patch.object(trainer_mock, "fit", return_value=None)
    mocker.patch("lightning.Trainer", return_value=trainer_mock)
    mock_mnasnet_trainer.train()
    mocker.patch("torch.save", side_effect=BaseException)
    with pytest.raises(BaseException):
        mock_mnasnet_trainer.train()


def test_search(mock_mnasnet_trainer, mocker):
    trainer_mock = mocker.Mock()
    mocker.patch.object(trainer_mock, "fit", return_value=None)
    mocker.patch("lightning.Trainer", return_value=trainer_mock)
    mock_mnasnet_trainer.search()


def test_save(mock_mnasnet_trainer, mocker):
    mocker.patch("pickle.dump", return_value=None)
    mock_mnasnet_trainer.save()


def test_get_valid_acc(mock_mnasnet_trainer, mocker):
    mock_mnasnet_trainer._search_model = mocker.MagicMock()
    mock_mnasnet_trainer.get_search_valid_acc()


def test_is_global_zero(mock_mnasnet_trainer, mocker):
    mock_mnasnet_trainer._search_trainer = mocker.MagicMock()
    mock_mnasnet_trainer.is_global_zero
