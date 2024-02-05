import pytest
import torch
from torch.optim import SGD

from aiaccel.nas.batch_dependent_lr_scheduler.batch_dependent_linear_lr_scheduler import BatchDependentLinearLRScheduler


class TestBatchDependentLinearLRScheduler:
    @pytest.fixture()
    def mock_optimizer(self):
        model = torch.nn.Linear(1, 1)
        return SGD(model.parameters(), lr=0.1)

    @pytest.fixture()
    def scheduler(self, mock_optimizer):
        return BatchDependentLinearLRScheduler(
            optimizer=mock_optimizer,
            num_epochs=10,
            num_batches=100,
            start_factor=0.3,
            end_factor=1.0,
            total_epochs=5,
            warmup_epochs=5,
            h_size=3,
            last_batch_id=-1,
            last_epoch=-1,
            verbose=False,
        )

    def test_init(self, scheduler):
        assert scheduler.start_factor == 0.3
        assert scheduler.end_factor == 1.0
        assert scheduler.diff_factor == 0.7
        assert scheduler.total_epochs == 5
        assert scheduler.total_iter == 500

    def test_get_lr_after_warmup(self, scheduler):
        lr = scheduler.get_lr_after_warmup()
        assert len(lr) == 1
        assert lr[0] == 0.03

        scheduler.last_epoch = 1
        lr = scheduler.get_lr_after_warmup()
        assert len(lr) == 1
        assert lr[0] > 0.03

        scheduler.last_epoch = 6
        lr = scheduler.get_lr_after_warmup()
        assert len(lr) == 1
        assert lr[0] == 0.1
