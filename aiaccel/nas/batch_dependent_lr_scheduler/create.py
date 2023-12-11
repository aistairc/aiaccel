from __future__ import annotations

from typing import TYPE_CHECKING, Any

from aiaccel.nas.batch_dependent_lr_scheduler.batch_dependent_cosine_lr_scheduler import BatchDependentCosineLRScheduler
from aiaccel.nas.batch_dependent_lr_scheduler.batch_dependent_exponential_lr_scheduler import (
    BatchDependentExponentialLRScheduler,
)
from aiaccel.nas.batch_dependent_lr_scheduler.batch_dependent_linear_lr_scheduler import BatchDependentLinearLRScheduler
from aiaccel.nas.batch_dependent_lr_scheduler.batch_dependent_multi_step_lr_scheduler import (
    BatchDependentMultiStepLRScheduler,
)

if TYPE_CHECKING:
    from torch.optim import Optimizer

    from aiaccel.nas.batch_dependent_lr_scheduler.abstract_batch_dependent_lr_shceduler import (
        AbstractBatchDependentLRScheduler,
    )


def create_batch_dependent_lr_scheduler(
    optimizer: Optimizer,
    hyperparameters: dict[str, Any],
    num_epochs: int,
    num_batches: int,
) -> AbstractBatchDependentLRScheduler:
    if hyperparameters["scheduler"] == "MultiStep":
        scheduler = BatchDependentMultiStepLRScheduler(
            optimizer=optimizer,
            milestones=list(range(hyperparameters["milestone_start"], num_epochs, hyperparameters["milestone_step"])),
            num_epochs=num_epochs,
            num_batches=num_batches,
            gamma=hyperparameters["gamma"],
            warmup_epochs=hyperparameters["warmup_epochs"],
        )
    elif hyperparameters["scheduler"] == "Linear":
        scheduler = BatchDependentLinearLRScheduler(
            optimizer=optimizer,
            num_epochs=num_epochs,
            num_batches=num_batches,
            start_factor=hyperparameters["start_factor"],
            end_factor=hyperparameters["end_factor"],
            total_epochs=hyperparameters["total_epochs"],
            warmup_epochs=hyperparameters["warmup_epochs"],
        )
    elif hyperparameters["scheduler"] == "Exponential":
        scheduler = BatchDependentExponentialLRScheduler(
            optimizer=optimizer,
            num_epochs=num_epochs,
            num_batches=num_batches,
            gamma=hyperparameters["gamma"],
            warmup_epochs=hyperparameters["warmup_epochs"],
        )
    else:
        assert hyperparameters["scheduler"] == "Cosine"
        scheduler = BatchDependentCosineLRScheduler(
            optimizer=optimizer,
            num_epochs=num_epochs,
            num_batches=num_batches,
            warmup_epochs=hyperparameters["warmup_epochs"],
        )
    return scheduler
