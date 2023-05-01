from __future__ import annotations

from typing import Any

from aiaccel.scheduler import AbstractScheduler
from aiaccel.experimental.mpi.config import MpiConfig
from aiaccel.experimental.mpi.scheduler.job.model.mpi_model import MpiModel
from aiaccel.experimental.mpi.util.mpi import Mpi


class MpiScheduler(AbstractScheduler):
    """A scheduler class running on a mpi systems.

    """
    def __init__(self, options: dict[str, Any]) -> None:
        super().__init__(options)
        self.config = MpiConfig(self.config_path)

    def get_stats(self) -> None:
        super().get_stats()

        self.stats = []
        trial_id_list = [job.trial_id for job in self.jobs]

        for trial_id in Mpi.get_trial_id_list():
            if trial_id in trial_id_list:
                self.stats.append({'name': str(trial_id)})
            else:
                self.logger.warning(f'**** Unknown trial_id: {trial_id}')

    def parse_trial_id(self, trial_id: str) -> Any:
        return trial_id
        
    def create_model(self) -> MpiModel:
        """Creates model object of state machine.

        Returns:
            MpiModel: Model object.
        """
        return MpiModel()
