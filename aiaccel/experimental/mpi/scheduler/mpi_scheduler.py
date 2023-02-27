from aiaccel.scheduler.local_scheduler import LocalScheduler
from aiaccel.experimental.mpi.config import MpiConfig
from aiaccel.experimental.mpi.scheduler.job.model.mpi_model import MpiModel


class MpiScheduler(LocalScheduler):
    """A scheduler class running on a mpi systems.

    """
    def __init__(self, options: dict) -> None:
        super().__init__(options)
        self.config = MpiConfig(self.config_path)

    def create_model(self) -> MpiModel:
        """Creates model object of state machine.

        Returns:
            MpiModel: Model object.
        """
        return MpiModel()
