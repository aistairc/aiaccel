from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf.dictconfig import DictConfig

from aiaccel.common import (class_master, class_optimizer, class_scheduler,
                            module_type_master, module_type_optimizer,
                            module_type_scheduler)
from aiaccel.storage import Storage
from aiaccel.util import TrialId
from aiaccel.workspace import Workspace


class AbstractModule(object):
    """An abstract class for Master, Optimizer and Scheduler.

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
        dict_hp (Path): A path to hp directory.
        dict_lock (Path): A path to lock directory.
        dict_output (Path): A path to output directory.
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
        self.storage = Storage(self.workspace.path)
        self.trial_id = TrialId(self.config)
        # TODO: Separate the generator if don't want to affect randomness each other.
        self._rng = np.random.RandomState(self.seed)
        self.module_name = module_name

        self.storage.variable.register(
            process_name=self.module_name,
            labels=['native_random_state', 'numpy_random_state', 'state']
        )

    def update_each_state_count(self) -> None:
        """Updates hyperparameter counters for ready, runnning, and finished
        states.
        """
        self.hp_ready = self.storage.get_num_ready()
        self.hp_running = self.storage.get_num_running()
        self.hp_finished = self.storage.get_num_finished()

    def get_module_type(self) -> str | None:
        """Get this module type.

        Returns:
            str: Name of this module type.
        """

        if class_master in self.__class__.__name__:
            return module_type_master
        elif class_optimizer in self.__class__.__name__:
            return module_type_optimizer
        elif class_scheduler in self.__class__.__name__:
            return module_type_scheduler
        else:
            return None

    def check_finished(self) -> bool:
        """Checks whether all optimization finished.

        Returns:
            bool: True if all optimizations are finished.
        """
        self.hp_finished = self.storage.get_num_finished()

        if self.hp_finished >= self.config.optimize.trial_number:
            return True

        return False

    def print_dict_state(self) -> None:
        """Print hp(hyperparameter) directory states.

        Returns:
            None
        """
        self.logger.info(
            f'{self.hp_finished}/{self.config.optimize.trial_number}, '
            f'finished, '
            f'ready: {self.hp_ready}, '
            f'running: {self.hp_running}'
        )

    def set_logger(
        self,
        logger_name: str,
        logfile: Path,
        file_level: int,
        stream_level: int,
        module_type: str
    ) -> None:
        """Set a default logger options.

        Args:
            logger_name (str): A name of a logger.
            logfile (Path): A path to a log file.
            file_level (int): A logging level for a log file output. For
                example logging.DEBUG
            stream_level (int): A logging level for a stream output.
            module_type (str): A module type of a caller.

        Returns:
            None
        """
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(logfile, mode='w')
        fh_formatter = logging.Formatter(
            '%(asctime)s %(levelname)-8s %(filename)-12s line '
            '%(lineno)-4s %(message)s'
        )
        fh.setFormatter(fh_formatter)
        fh.setLevel(file_level)

        ch = logging.StreamHandler()
        ch_formatter = logging.Formatter(
            f'{module_type} %(levelname)-8s %(message)s'
        )
        ch.setFormatter(ch_formatter)
        ch.setLevel(stream_level)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def pre_process(self) -> None:
        """Pre-procedure before executing processes.

        Returns:
            None
        """
        raise NotImplementedError

    def post_process(self) -> None:
        """Post-procedure after executed processes.

        Returns:
            None

        Raises:
            NotImplementedError: Causes when he inherited class does not
                implement.
        """
        raise NotImplementedError

    def inner_loop_main_process(self) -> bool:
        """A main loop process. This process is repeated every main loop.

        Returns:
            None

        Raises:
            NotImplementedError: Causes when the inherited class does not
                implement.
        """
        raise NotImplementedError

    def _serialize(self, trial_id: int) -> None:
        """Serialize this module.

        Returns:
            None
        """
        self.storage.variable.d['state'].set(trial_id, self)

        # random state
        self.storage.variable.d['numpy_random_state'].set(trial_id, self.get_numpy_random_state())

    def _deserialize(self, trial_id: int) -> None:
        """ Deserialize this module.

        Returns:
            None
        """
        self.__dict__.update(self.storage.variable.d['state'].get(trial_id).__dict__.copy())

        # random state
        self.set_numpy_random_state(self.storage.variable.d['numpy_random_state'].get(trial_id))

    def write_random_seed_to_debug_log(self) -> None:
        """Writes the random seed to the logger as debug information.
        """
        self.logger.debug(f'create numpy random generator by seed: {self.seed}')

    def get_numpy_random_state(
        self
    ) -> dict[str, Any] | tuple[str, np.ndarray[Any, np.dtype[np.uint32]], int, int, float]:
        """Gets random state.

        Returns:
            dict[str, Any] | tuple[str, ndarray[Any, dtype[uint32]], int, int, float]: A tuple representing the
                internal state of the generator if legacy is True. If legacy is False, or the BitGenerator is not
                MT19937, then state is returned as a dictionary.
        """
        return self._rng.get_state()

    def set_numpy_random_state(
        self,
        state: Any
    ) -> None:
        """Gets random state.

        Args:
            state (dict[str, Any] | tuple[str, ndarray[Any, np.dtype[uint32]], int, int, float]): A tuple or dictionary
                representing the internal state of the generator.
        """
        self._rng.set_state(state)

    def check_error(self) -> bool:
        """ Check to confirm if an error has occurred.

        Args:
            None

        Returns:
            bool: True if no error, False if with error.
        """
        return True

    def resume(self) -> None:
        """ When in resume mode, load the previous
                optimization data in advance.

        Args:
            None

        Returns:
            None
        """
        if (
            self.config.resume is not None and
            self.config.resume > 0
        ):
            self._deserialize(self.config.resume)

    def __getstate__(self) -> dict[str, Any]:
        obj = self.__dict__.copy()
        del obj['storage']
        del obj['config']
        return obj
