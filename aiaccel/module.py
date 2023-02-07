from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

import aiaccel
from aiaccel.config import Config
from aiaccel.storage.storage import Storage
from aiaccel.util.trialid import TrialId


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
        dict_log (Path): A path to log directory.
        dict_output (Path): A path to output directory.
        dict_runner (Path): A path to runner directory.
        dict_verification (Path): A path to verification directory.
        hp_finished (int): A number of files in hp/finished directory.
        hp_ready (int): A number of files in hp/ready directory.
        hp_running (int): A number of files in hp/running directory.
        logger (logging.Logger): A logger object.
        loop_count (int): A loop count that is incremented in loop method.
    """

    def __init__(self, options: dict[str, str | int | bool]) -> None:
        # === Load config file===
        self.options = options
        self.config_path = Path(self.options['config']).resolve()
        self.config = Config(self.config_path)
        self.ws = Path(self.config.workspace.get()).resolve()

        # working directory
        self.dict_alive = self.ws / aiaccel.dict_alive
        self.dict_hp = self.ws / aiaccel.dict_hp
        self.dict_lock = self.ws / aiaccel.dict_lock
        self.dict_log = self.ws / aiaccel.dict_log
        self.dict_output = self.ws / aiaccel.dict_output
        self.dict_result = self.ws / aiaccel.dict_result
        self.dict_runner = self.ws / aiaccel.dict_runner
        self.dict_verification = self.ws / aiaccel.dict_verification
        self.dict_hp_ready = self.ws / aiaccel.dict_hp_ready
        self.dict_hp_running = self.ws / aiaccel.dict_hp_running
        self.dict_hp_finished = self.ws / aiaccel.dict_hp_finished
        self.dict_storage = self.ws / aiaccel.dict_storage

        # alive file
        self.alive_master = self.dict_alive / aiaccel.alive_master
        self.alive_optimizer = self.dict_alive / aiaccel.alive_optimizer
        self.alive_scheduler = self.dict_alive / aiaccel.alive_scheduler

        self.logger = None
        self.fh = None
        self.ch = None
        self.ch_formatter = None
        self.loop_count = 0
        self.hp_ready = 0
        self.hp_running = 0
        self.hp_finished = 0
        self.seed = self.config.randseed.get()
        self.storage = Storage(self.ws)
        self.trial_id = TrialId(self.options['config'])
        # TODO: Separate the generator if don't want to affect randomness each other.
        self._rng: np.random.RandomState | None = None

        self.storage.variable.register(
            process_name=self.options['process_name'],
            labels=['native_random_state', 'numpy_random_state', 'state']
        )

    def get_each_state_count(self) -> None:
        """Updates the number of files in hp(hyper parameter) directories.

        Returns:
            None
        """
        self.hp_ready = self.storage.get_num_ready()
        self.hp_running = self.storage.get_num_running()
        self.hp_finished = self.storage.get_num_finished()

    def get_module_type(self) -> str | None:
        """Get this module type.

        Returns:
            str: Name of this module type.
        """

        if aiaccel.class_master in self.__class__.__name__:
            return aiaccel.module_type_master
        elif aiaccel.class_optimizer in self.__class__.__name__:
            return aiaccel.module_type_optimizer
        elif aiaccel.class_scheduler in self.__class__.__name__:
            return aiaccel.module_type_scheduler
        else:
            return None

    def check_finished(self) -> bool:
        """Check whether all optimization finished or not.

        Returns:
            bool: All optimization finished or not.
        """
        self.hp_finished = self.storage.get_num_finished()

        if self.hp_finished >= self.config.trial_number.get():
            return True

        return False

    def print_dict_state(self) -> None:
        """Print hp(hyperparameter) directory states.

        Returns:
            None
        """
        self.logger.info(
            f'{self.hp_finished}/{self.config.trial_number.get()}, '
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
        fh_formatter = (
            '%(asctime)s %(levelname)-8s %(filename)-12s line '
            '%(lineno)-4s %(message)s'
        )
        fh_formatter = logging.Formatter(fh_formatter)
        fh.setFormatter(fh_formatter)
        fh.setLevel(file_level)

        ch = logging.StreamHandler()
        ch_formatter = (f'{module_type} %(levelname)-8s %(message)s')
        ch_formatter = logging.Formatter(ch_formatter)
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

    def set_numpy_random_seed(self) -> None:
        """ set any random seed.

        Args:
            None

        Returns:
            None
        """
        self.logger.debug(f'set numpy random seed: {self.seed}')
        if self._rng is None:
            self.create_numpy_random_generator()
        np.random.set_state(self.get_numpy_random_state())

    def create_numpy_random_generator(self) -> None:
        """ create random generator using any random seed.

        Args:
            None

        Returns:
            None
        """
        self.logger.debug(f'create numpy random generator by seed: {self.seed}')
        self._rng = np.random.RandomState(self.seed)

    def get_numpy_random_state(self) -> tuple:
        """ get random state.

        Args:
            None

        Returns:
            numpy.random.get_state (tuple)
        """
        return self._rng.get_state()

    def set_numpy_random_state(self, state: tuple) -> None:
        """ get random state.

        Args:
            state (tuple): random state

        Returns:
            None
        """
        self._rng.set_state(state)

    def check_error(self) -> bool:
        """ Check to confirm if an error has occurred.

        Args:
            None

        Returns:
            True: no error | False: with error.
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
            self.options['resume'] is not None and
            self.options['resume'] > 0
        ):
            self._deserialize(self.options['resume'])

    def __getstate__(self):
        obj = self.__dict__.copy()
        del obj['storage']
        del obj['config']
        del obj['options']
        return obj
