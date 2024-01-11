from __future__ import annotations

from aiaccel.manager.job.model.mpi_model import MpiModel
from aiaccel.manager.local_manager import LocalManager


class MpiManager(LocalManager):
    """A manager class running on a mpi systems."""

    def create_model(self) -> MpiModel:
        """Creates model object of state machine.

        Returns:
            MpiModel: Model object.
        """
        return MpiModel()
