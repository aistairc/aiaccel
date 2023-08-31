from __future__ import annotations

from aiaccel.scheduler.abstract_scheduler import AbstractScheduler
from aiaccel.scheduler.job.model.mpi_model import MpiModel


class MpiScheduler(AbstractScheduler):
    """A scheduler class running on a mpi systems."""

    def create_model(self) -> MpiModel:
        """Creates model object of state machine.

        Returns:
            MpiModel: Model object.
        """
        return MpiModel()
