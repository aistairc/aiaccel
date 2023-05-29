from __future__ import annotations

from aiaccel.experimental.mpi.scheduler.job.model.mpi_model import MpiModel
from aiaccel.scheduler.abstract_scheduler import AbstractScheduler


class MpiScheduler(AbstractScheduler):
    """A scheduler class running on a mpi systems."""

    def create_model(self) -> MpiModel:
        """Creates model object of state machine.

        Returns:
            MpiModel: Model object.
        """
        return MpiModel()
