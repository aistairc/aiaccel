from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf.dictconfig import DictConfig

from aiaccel.storage import Storage
from aiaccel.util import ColoredHandler, TrialId, str_to_logging_level
from aiaccel.workspace import Workspace


class AbstractModule(object):
    """An abstract class for Optimizer and Manager.

    The procedure of this class is as follows:

    1. At first, deserialize() is called.
    2. start() is called.
    3. pre_process() is called.
    4. loop() is called.

        | 4-1. in while loop, inner_loop_main_process() is called.
        | 4-2. in while loop, loop_count is incremented.

    5. call post_process()

     Args:
        options (dict[str, str | int | bool]): A dictionary containing
            command line options.

    Attributes:
        options (dict[str, str | int | bool]): A dictionary containing
            command line options.
        config_path (Path): Path to the configuration file.
        config (Config): A config object.
        ws (Path): A path to a current workspace.
        dict_lock (Path): A path to lock directory.
        dict_runner (Path): A path to runner directory.
        hp_finished (int): A number of files in hp/finished directory.
        hp_ready (int): A number of files in hp/ready directory.
        hp_running (int): A number of files in hp/running directory.
        logger (logging.Logger): A logger object.
        loop_count (int): A loop count that is incremented in loop method.
    """

    def __init__(self, config: DictConfig, module_name: str) -> None:
        self.config = config
        self.workspace = Workspace(self.config.generic.workspace)
        self.goals = [item.value for item in self.config.optimize.goal]
        self.logger: Any = None
        self.fh: Any = None
        self.ch: Any = None
        self.ch_formatter: Any = None
        self.loop_count = 0
        self.hp_ready = 0
        self.hp_running = 0
        self.hp_finished = 0
        self.seed = self.config.optimize.rand_seed
        self.storage = Storage(self.workspace.storage_file_path)
        self.trial_id = TrialId(self.config)
        # TODO: Separate the generator if don't want to affect randomness each other.
        self._rng = np.random.RandomState(self.seed)
        self.module_name = module_name

        self.storage.variable.register(
            process_name=self.module_name, labels=["native_random_state", "numpy_random_state", "state"]
        )

    def set_config(self, config: DictConfig) -> None:
        self.config = config

    def set_storage(self, storage: Storage) -> None:
        self.storage = storage
        self.storage.variable.register(
            process_name=self.module_name, labels=["native_random_state", "numpy_random_state", "state"]
        )

    def set_logger(self, logger_name: str, logfile: Path, file_level: str, stream_level: str) -> None:
        """Set a default logger options.

        Args:
            logger_name (str): A name of a logger.
            logfile (Path): A path to a log file.
            file_level (str): A logging level for a log file output. For
                example logging.DEBUG
            stream_level (str): A logging level for a stream output.
        Returns:
            None
        """
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(logfile, mode="w")
        fh_formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(filename)-12s line " "%(lineno)-4s %(message)s")
        fh.setFormatter(fh_formatter)
        fh.setLevel(str_to_logging_level(file_level))

        ch = ColoredHandler(sys.stdout)
        ch_formatter = logging.Formatter("%(levelname)-8s %(message)s")
        ch.setFormatter(ch_formatter)
        ch.setLevel(str_to_logging_level(stream_level))

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def serialize(self, trial_id: int) -> None:
        """Serialize this module.

        Returns:
            None
        """
        self.storage.variable.d["state"].set(trial_id, self)

        # random state
        self.storage.variable.d["numpy_random_state"].set(trial_id, self.get_numpy_random_state())

    def deserialize(self, trial_id: int) -> None:
        """Deserialize this module.

        Returns:
            None
        """
        __dict__ = self.storage.variable.d["state"].get(trial_id).__dict__.copy()
        self.logger.debug(f"deserialize {self.__class__.__name__} module:")
        self.logger.debug(f"  {__dict__}")
        self.__dict__.update(__dict__)

        # random state
        _random_state = self.storage.variable.d["numpy_random_state"].get(trial_id)
        self.set_numpy_random_state(_random_state)
        self.logger.debug("deserialize random state")
        self.logger.debug(f"{_random_state}")

    def write_random_seed_to_debug_log(self) -> None:
        """Writes the random seed to the logger as debug information."""
        self.logger.debug(f"create numpy random generator by seed: {self.seed}")

    def get_numpy_random_state(
        self,
    ) -> dict[str, Any] | tuple[str, np.ndarray[Any, np.dtype[np.uint32]], int, int, float]:
        """Gets random state.

        Returns:
            dict[str, Any] | tuple[str, ndarray[Any, dtype[uint32]], int, int, float]: A tuple representing the
                internal state of the generator if legacy is True. If legacy is False, or the BitGenerator is not
                MT19937, then state is returned as a dictionary.
        """
        return self._rng.get_state()

    def set_numpy_random_state(self, state: Any) -> None:
        """Gets random state.

        Args:
            state (dict[str, Any] | tuple[str, ndarray[Any, np.dtype[uint32]], int, int, float]): A tuple or dictionary
                representing the internal state of the generator.
        """
        self._rng.set_state(state)

    def is_error_free(self) -> bool:
        """Check if there has been an error.

        This method should be implemented by subclasses to define how to check for errors.

        Returns:
            bool: True if there has been no error, False otherwise.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        return True

    def resume(self) -> None:
        """Load previous optimization data when in resume mode.

        This method should be implemented by subclasses to define how to load previously saved optimization data.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError

    def __getstate__(self) -> dict[str, Any]:
        """Prepare the object for serialization.

        Certain attributes may need to be removed or modified before serialization.

        Returns:
            dict[str, Any]: A dictionary that can be serialized.
        """
        obj = self.__dict__.copy()
        del obj["storage"]
        del obj["config"]
        return obj
