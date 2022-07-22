from pathlib import Path
import aiaccel
from aiaccel.storage.model.fs import Datalist
from aiaccel.storage.model.fs import _timestamp
from aiaccel.config import Config


class TimeStamp:
    def __init__(self, config: Config):
        self.config = config
        self.workspace = Path(config.workspace.get()).resolve()
        self.name_length = config.name_length.get()
        self.file_type = "yaml"
        self.path = self.workspace / aiaccel.dict_timestamp
        self.datas = Datalist()
        self.default = {
            'start_time': None,
            'end_time': None
        }

    def get_file_list(self) -> list:
        return list(self.path.glob(f"*.{self.file_type}"))

    def add(self, trial_id):
        self.datas.add(
            trial_id,
            _timestamp(self.config, trial_id)
        )

    def get(self, trial_id: int) -> dict:
        return self.datas.get(trial_id)

    def set(self, trial_id: int, contents: dict) -> None:
        self.datas.set(trial_id, contents)

    def update(self):
        self.datas.clear()
        paths = self.get_file_list()
        for path in paths:
            trial_id = int(path.stem)
            self.add(trial_id)

    def set_any_trial_start_time(self, trial_id: int, start_time: str) -> None:
        """Set the specified start time for the specified trial.

        Args:
            trial_id (int) : Any trial id
            start_time(str): "MM/DD/YYYY hh:mm:ss"

        Returns:
            None
        """
        self.update()
        self.add(trial_id)
        timestamp = self.datas.get(trial_id)
        if timestamp is None:
            timestamp = self.default
        timestamp['start_time'] = start_time
        self.datas.set(trial_id, timestamp)

    def set_any_trial_end_time(self, trial_id: int, end_time: str) -> None:
        """Set the specified end time for the specified trial.

        Args:
            trial_id(int): Any trial id
            end_time(str): "MM/DD/YYYY hh:mm:ss"

        Returns:
            None
        """
        self.update()
        self.add(trial_id)
        timestamp = self.datas.get(trial_id)
        if timestamp is None:
            timestamp = self.default
        timestamp['end_time'] = end_time
        self.datas.set(trial_id, timestamp)

    def get_any_trial_start_time(self, trial_id: int) -> str:
        """Obtains the start time of the specified trial.

        Args:
            trial_id(int): Any trial id

        Returns:
            start_time(str): "MM/DD/YYYY hh:mm:ss"
        """
        self.update()
        timestamp = self.datas.get(trial_id)
        if timestamp is None:
            return None
        return timestamp['start_time']

    def get_any_trial_end_time(self, trial_id: int) -> str:
        """Obtains the end time of the specified trial.

        Args:
            trial_id(int): Any trial id

        Returns:
            end_time(str): "MM/DD/YYYY hh:mm:ss"
        """
        self.update()
        timestamp = self.datas.get(trial_id)
        if timestamp is None:
            return None
        return timestamp['end_time']

    def all_delete(self) -> None:
        """Clear table

        Returns:
            None
        """
        self.update()
        for d in self.datas.data:
            d.remove()
