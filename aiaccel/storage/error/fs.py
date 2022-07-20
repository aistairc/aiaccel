from typing import Union
from pathlib import Path
import aiaccel
from aiaccel.config import Config
from aiaccel.storage.model.fs import Datalist
from aiaccel.storage.model.fs import _error


class Error:
    def __init__(self, config: Config):
        self.config = config
        self.workspace = Path(config.workspace.get()).resolve()
        self.name_length = config.name_length.get()
        self.path = self.workspace / aiaccel.dict_error
        self.file_time = "yaml"
        self.errors = Datalist()

    def get_file_list(self) -> list:
        return sorted(list(self.path.glob(f"*.{self.file_time}")))

    def clear(self):
        self.errors.clear()

    def add(self, trial_id):
        self.errors.add(
            trial_id,
            _error(self.config, trial_id)
        )

    def set(self, trial_id, error: str):
        self.errors.set(trial_id, error)

    def update(self):
        paths = self.get_file_list()
        for path in paths:
            trial_id = int(path.stem)
            self.add(trial_id)

    def set_any_trial_error(self, trial_id: int, error_message: str) -> None:
        """Set any error message for any trial.

        Args:
            trial_id (int): Any trial id
            error_message(str): Any error message

        Returns:
            None
        """
        self.update()
        self.add(trial_id)
        self.set(trial_id, error_message)

    def get_any_trial_error(self, trial_id: int) -> Union[None, str]:
        """Get error messages for any trial.

        Args:
            trial_id (int): Any trial id

        Returns:
            str | None
        """
        self.update()
        return self.errors.get(trial_id)

    def get_error_trial_id(self) -> list:
        """Obtain a list of trial ids in which an error occurred.

        Returns:
            trial_ids(list): trial id list
        """
        filenames = self.get_file_list()
        trial_id = [int(filename.stem) for filename in filenames]
        return trial_id

    def all_delete(self) -> None:
        self.update()
        for d in self.errors.data:
            d.remove()
