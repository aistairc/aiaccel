from __future__ import annotations

from aiaccel.manager.abstract_manager import AbstractManager
from aiaccel.manager.job.model.mpi_model import MpiModel


class MpiManager(AbstractManager):
    """A manager class running on a mpi systems."""

    def create_model(self) -> MpiModel:
        """Creates model object of state machine.

        Returns:
            MpiModel: Model object.
        """
        return MpiModel()
