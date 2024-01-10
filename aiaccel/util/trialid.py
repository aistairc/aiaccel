from __future__ import annotations

from typing import Any

import fasteners
from omegaconf.dictconfig import DictConfig

from aiaccel.common import file_hp_count, file_hp_count_lock, file_hp_count_lock_timeout
from aiaccel.workspace import Workspace


class TrialId:
    """Provides interface to current trial id.
    Args:
        config_path (str): Path to the config file.
    Attributes:
        config (Config): Config object.
        ws (Path): Path to the workspace.
        name_length (int): Name length of hp files.
        file_hp_count_fmt (str): String format of hp file name.
        count_path (Path): Path to "count.txt" containing current trial id,
            i.e. `ws`/hp/count.txt.
        lock_path (Path): Path to "count.lock", i.e. `ws`/hp/count.lock.
        lock (fasteners.InterProcessLock): An interprocess lock.
    """

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.workspace = Workspace(self.config.generic.workspace)
        self.name_length = self.config.job_setting.name_length
        self.file_hp_count_fmt = f"%0{self.name_length}d"

        self.count_path = self.workspace.path / file_hp_count
        self.lock_path = self.workspace.path / file_hp_count_lock
        self.lock = fasteners.InterProcessLock(str(self.lock_path))

        if self.get() is None:
            self.initial()

    def zero_padding_any_trial_id(self, trial_id: int) -> str:
        """Returns string of trial id padded by zeros.
        Args:
            trial_id (int): Trial id.
        Returns:
            str: Trial id padded by zeros.
        """
        return self.file_hp_count_fmt % trial_id

    def increment(self) -> None:
        """Increments trial id."""
        if self.lock.acquire(timeout=file_hp_count_lock_timeout):
            trial_id = 0
            if self.count_path.exists():
                trial_id = int(self.count_path.read_text())
                trial_id += 1
            self.count_path.write_text("%d" % trial_id)
            self.lock.release()

    def get(self) -> Any:
        """Returns trial id.
        Returns:
            int | None: Trial id. None if count.txt does not exist.
        """
        if self.count_path.exists() is False:
            return None
        return int(self.count_path.read_text())

    def initial(self, num: int = 0) -> None:
        """Initialize trial id.
        Args:
            num (int, optional): _description_. Defaults to 0.
        """
        if self.lock.acquire(timeout=file_hp_count_lock_timeout):
            trial_id = num
            self.count_path.write_text("%d" % trial_id)
            self.lock.release()

    @property
    def integer(self) -> Any:
        """Trial id."""
        return self.get()

    @property
    def string(self) -> str:
        """Formatted trial id."""
        return self.file_hp_count_fmt % self.get()
