from pathlib import Path
from typing import Union

import fasteners

import aiaccel
from aiaccel.config import load_config



class TrialId:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path).resolve()
        self.config = load_config(str(self.config_path))

        self.ws = Path(self.config.generic.workspace).resolve()
        self.name_length = self.config.job_setting.name_length
        self.file_hp_count_fmt = f'%0{self.name_length}d'
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
