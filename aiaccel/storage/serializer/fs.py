from typing import Any
from pathlib import Path
from aiaccel.util.snapshot import SnapShot
from aiaccel.util.retry import retry
from aiaccel.config import Config


class Serializer():
    def __init__(self, config: Config):
        self.workspace = Path(config.workspace.get()).resolve()
        self.name_length = config.name_length.get()
        self.snapshot = SnapShot(self.workspace)

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def set_any_trial_serialize(
        self,
        trial_id: int,
        optimization_variable,
        process_name: str,
        native_random_state: tuple,
        numpy_random_state: tuple
    ) -> None:
        """Sets serialization data for a given trial.

        Args:
            trial_id (int): Any trial id
            optimization_variable: serialized data
            process_name (str): master, optimizer, scheduler

        Returns:
            None
        """
        self.snapshot.save(
            trial_id,
            process_name,
            optimization_variable,
            native_random_state,
            numpy_random_state
        )

    @retry(_MAX_NUM=60, _DELAY=1.0)
    def get_any_trial_serialize(self, trial_id: int, process_name: str) -> Any:
        """Obtain serialized data for a given trial.

        Args:
            trial_id (int): Any trial id
            process_name (str): master, optimizer, scheduler

        Returns:
            serialized data
        """
        if self.snapshot.load(trial_id, process_name) is False:
            return None

        return (
            self.snapshot.optimization_variables,
            self.snapshot.random_state_native,
            self.snapshot.random_state_numpy
        )

    def delete_any_trial_serialize(self, trial_id: int) -> None:
        self.snapshot.delete(trial_id)

    def is_exists_any_trial(self, trial_id: int):
        process_names = [
            'master',
            'optimizer',
            'scheduler'
        ]
        for process_name in process_names:
            if self.snapshot.load(trial_id, process_name) is False:
                return False
        return True
