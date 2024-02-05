import pytest
import torch
from torch.optim import SGD

from aiaccel.nas.batch_dependent_lr_scheduler.batch_dependent_exponential_lr_scheduler import (
    BatchDependentExponentialLRScheduler,
)


class TestBatchDependentExponentialLRScheduler:
    @pytest.fixture()
    def mock_optimizer(self):
        model = torch.nn.Linear(1, 1)
        return SGD(model.parameters(), lr=0.1)

    @pytest.fixture()
    def scheduler(self, mock_optimizer):
        return BatchDependentExponentialLRScheduler(
            optimizer=mock_optimizer,
            num_epochs=10,
            num_batches=100,
            gamma=0.9,
            warmup_epochs=5,
            h_size=3,
            last_batch_id=-1,
            last_epoch=-1,
            verbose=False,
        )

    def test_init(self, scheduler):
        assert scheduler.gamma == 0.9 ** (1 / 100)

    def test_get_lr_after_warmup(self, scheduler):
        lr = scheduler.get_lr_after_warmup()
        assert len(lr) == 1
        assert lr[0] == 0.1

        scheduler.last_epoch = 1
        lr = scheduler.get_lr_after_warmup()
        assert len(lr) == 1
        assert lr[0] == 0.1 * scheduler.gamma
