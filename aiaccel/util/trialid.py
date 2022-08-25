from pathlib import Path
from aiaccel.config import Config
import aiaccel
import fasteners
from typing import Union

# wd/


class TrialId:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path).resolve()
        self.config = Config(str(self.config_path))

        self.ws = Path(self.config.workspace.get()).resolve()
        self.name_length = self.config.name_length.get()
        self.file_hp_count_fmt = '%0{}d'.format(self.name_length)
        self.dict_hp = self.ws / aiaccel.dict_hp

        self.count_path = self.dict_hp / aiaccel.file_hp_count
        self.lock_path = self.dict_hp / aiaccel.file_hp_count_lock
        self.lock = fasteners.InterProcessLock(str(self.lock_path))

        if self.get() is None:
            self.initial()

    def zero_padding_any_trial_id(self, trial_id: int):
        return self.file_hp_count_fmt % trial_id

    def increment(self):
        if self.lock.acquire(timeout=aiaccel.file_hp_count_lock_timeout):
            trial_id = 0
            if self.count_path.exists():
                trial_id = int(self.count_path.read_text())
                trial_id += 1
            self.count_path.write_text('%d' % trial_id)
            self.lock.release()

    def get(self) -> Union[None, int]:
        if self.count_path.exists() is False:
            return None
        return int(self.count_path.read_text())

    def initial(self, num: int = 0) -> None:
        if self.lock.acquire(timeout=aiaccel.file_hp_count_lock_timeout):
            trial_id = num
            self.count_path.write_text('%d' % trial_id)
            self.lock.release()

    @property
    def integer(self) -> None:
        return self.get()

    @property
    def string(self) -> None:
        return self.file_hp_count_fmt % self.get()
