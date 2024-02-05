from unittest.mock import MagicMock

import pytest
import torch
from torch.optim import SGD

from aiaccel.nas.batch_dependent_lr_scheduler.abstract_batch_dependent_lr_shceduler import (
    AbstractBatchDependentLRScheduler,
)


class TestAbstractBatchDependentLRScheduler:
    @pytest.fixture()
    def mock_optimizer(self):
        model = torch.nn.Linear(1, 1)
        return SGD(model.parameters(), lr=0.1)

    @pytest.fixture()
    def scheduler(self, mock_optimizer):
        scheduler = AbstractBatchDependentLRScheduler(
            optimizer=mock_optimizer,
            num_epochs=10,
            num_batches=100,
            warmup_epochs=5,
            h_size=3,
            last_batch_id=-1,
            last_epoch=-1,
            verbose=False,
        )
        scheduler.get_lr_after_warmup = MagicMock(return_value=[0.1, 0.2, 0.3])
        return scheduler

    def test_init(self, scheduler):
        assert scheduler._initial_batch_id == -1
        assert scheduler.num_epochs == 10
        assert scheduler.num_batches == 100
        assert scheduler.warmup_epochs == 5
        assert scheduler.h_size == 3
        assert scheduler.last_batch_id == -1
        assert scheduler.last_epoch == 0

    def test_get_lr(self, scheduler):
        assert scheduler.get_lr() == [0.03333333333333333]

    def test_step(self, scheduler):
        scheduler.step()
        assert scheduler.last_batch_id == -1
        assert scheduler.last_epoch == 1

    def test_is_before_warmup(self, scheduler):
        assert scheduler.is_before_warmup() == True

    def test_get_lr_before_warmup(self, scheduler):
        assert scheduler.get_lr_before_warmup() == [0.03333333333333333]

    def test_batch_step(self, scheduler):
        scheduler.batch_step()
        assert scheduler.last_batch_id == 0
