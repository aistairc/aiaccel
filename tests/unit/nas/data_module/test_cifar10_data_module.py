from pathlib import Path
from typing import Any

import pytest
from torchvision.datasets import CIFAR10

from aiaccel.nas.data_module.cifar10_data_module import Cifar10DataModule


class TestCifar10DataModule:
    @pytest.fixture(scope="class", autouse=True)
    def data_module(self, class_mocker):
        dm = class_mocker.Mock(spec=Cifar10DataModule)
        dm.data_dir = Path()
        dm.train_batch_size = 32
        dm.test_batch_size = 32
        dm.num_workers = 1
        dm.persistent_workers = True
        dm.dims = (3, 32, 32)
        dm.num_classes = 10
        dm.num_train_data = 0
        dm.num_test_data = 0

        return dm

    def test_init(self, data_module):
        assert data_module.data_dir.is_dir()
        assert data_module.train_batch_size == 32
        assert data_module.test_batch_size == 32
        assert data_module.num_workers == 1
        assert data_module.persistent_workers == True
        assert data_module.dims == (3, 32, 32)
        assert data_module.num_classes == 10
        assert data_module.num_train_data == 0
        assert data_module.num_test_data == 0

    def test_prepare_data(self, mocker):
        class Cifar10Train:
            def __len__(self):
                return 50000

        class Cifar10Test:
            def __len__(self):
                return 10000

        def cm_init(data_dir, train, download):
            if train:
                return Cifar10Train()
            return Cifar10Test()

        mocker.patch("aiaccel.nas.data_module.cifar10_data_module.CIFAR10", side_effect=cm_init)
        dm = Cifar10DataModule(data_dir=Path(), train_batch_size=32, test_batch_size=32, num_workers=1)
        dm.prepare_data()
        assert dm.num_train_data == 50000
        assert dm.num_test_data == 10000

    def test_setup(self, mocker):
        class Cifar10Train:
            def __len__(self):
                return 45000

        class Cifar10Val:
            def __len__(self):
                return 5000

        mocker.patch("aiaccel.nas.data_module.cifar10_data_module.CIFAR10", return_value=Cifar10Train())
        mocker.patch(
            "aiaccel.nas.data_module.cifar10_data_module.random_split",
            return_value=(Cifar10Train(), Cifar10Val()),
        )

        dm = Cifar10DataModule(data_dir=Path(), train_batch_size=32, test_batch_size=32, num_workers=1)
        dm.prepare_data()
        dm.setup(stage="fit")
        assert len(dm.cifar_train) == 45000
        assert len(dm.cifar_val) == 5000

        dm.setup(stage="test")
        assert len(dm.cifar_test) == 45000

    def test_dataloaders(self, mocker):
        class Cifar10Datasets:
            def __len__(self):
                return 45000

        class Cifar10DataLoader:
            def __init__(self):
                self.dataset = Cifar10Datasets()

        dm = Cifar10DataModule(data_dir=Path(), train_batch_size=32, test_batch_size=32, num_workers=1)
        dm.cifar_train = None
        dm.cifar_val = None
        dm.cifar_test = None
        mocker.patch("aiaccel.nas.data_module.cifar10_data_module.DataLoader", return_value=Cifar10DataLoader())
        train_dataloader = dm.train_dataloader()
        assert len(train_dataloader.dataset) == 45000

        val_dataloader = dm.val_dataloader()
        assert len(val_dataloader.dataset) == 45000

        test_dataloader = dm.test_dataloader()
        assert len(test_dataloader.dataset) == 45000
