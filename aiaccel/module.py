import aiaccel
import logging
import os
import time
import sys
from aiaccel.config import Config
from aiaccel.util.process import is_process_running
from pathlib import Path
import numpy as np
import random
from aiaccel.storage.storage import Storage
from aiaccel.util.trialid import TrialId


class AbstractModule(object):
    """An abstract class for Master, Optimizer and Scheduler.

    The procedure of this class is as follows:
        1. At first, deserialize() is called.
        2. start() is called.
        3. pre_process() is called.
        4. loop() is called.
            4-1. loop() calls loop_pre_process().
            4-2. in while loop, inner_loop_pre_process() is called.
            4-3. in while loop, inner_loop_main_process() is called.
            4-4. in while loop, inner_loop_post_process() is called.
            4-5. in while loop, loop_count is incremented.
            4-6. in while loop, serialize() is called.
            4-7. loop() calls loop_post_process().
        5. call post_process()

    Attributes:
        config (ConfileWrapper): A config object.
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
        ws (Path): A path to a current workspace.
    """

    def __init__(self, options: dict) -> None:
        """
        Args:
            config (str): A file name of a configuration.
        """
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
        self.ch_formatter = None
        self.loop_count = 0
        self.hp_ready = 0
        self.hp_running = 0
        self.hp_finished = 0
        self.sleep_time = 1.0
        self.seed = self.config.randseed.get()
        self.storage = Storage(
            self.ws,
            fsmode=options['fs'],
            config_path=self.config.config_path
        )
        self.trial_id = TrialId(self.options['config'])
        self.serialize_datas = {}
        self.deserialize_datas = {}

        self.process_names = [
            aiaccel.module_type_master,
            aiaccel.module_type_optimizer,
            aiaccel.module_type_scheduler
        ]

    def get_each_state_count(self) -> None:
        """Updates the number of files in hp(hyper parameter) directories.

        Returns:
            None
        """
        self.hp_ready = self.storage.get_num_ready()
        self.hp_running = self.storage.get_num_running()
        self.hp_finished = self.storage.get_num_finished()

    def get_module_type(self) -> str:
        """Get this module type.

        Returns:
            str: This module type(name).
        """

        if aiaccel.class_master in self.__class__.__name__:
            return aiaccel.module_type_master
        elif aiaccel.class_optimizer in self.__class__.__name__:
            return aiaccel.module_type_optimizer
        elif aiaccel.class_scheduler in self.__class__.__name__:
            return aiaccel.module_type_scheduler
        else:
            return None

    def get_alive_file(self) -> Path:
        if aiaccel.class_master in self.__class__.__name__:
            return self.alive_master
        elif aiaccel.class_optimizer in self.__class__.__name__:
            return self.alive_optimizer
        elif aiaccel.class_scheduler in self.__class__.__name__:
            return self.alive_scheduler
        else:
            self.logger.error(f'Unknown type of module: {self.__class__.__name__}')
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

    def exit_alive(self, process_name: str) -> None:
        """Exit the execution.

        Args:
            filename (Path): A path to an alive file.

        Returns:
            None
        """
        self.storage.alive.stop_any_process(process_name)

    def print_dict_state(self) -> None:
        """Print hp(hyperparameter) directory states.

        Returns:
            None
        """
        self.logger.info('{}/{} finished, ready: {}, running: {}'.format(
            self.hp_finished,
            self.config.trial_number.get(),
            self.hp_ready,
            self.hp_running)
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
            file_level (int): A logging level for a log file output. For example logging.DEBUG
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
        ch_formatter = (
            '{} %(levelname)-8s %(message)s'.format(module_type)
        )
        ch_formatter = logging.Formatter(ch_formatter)
        ch.setFormatter(ch_formatter)
        ch.setLevel(stream_level)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def remove_logger_handler(self):
        self.logger = None

    def pre_process(self) -> None:
        """Pre-procedure before executing processes.

        Returns:
            None
        """
        module_type = self.get_module_type()

        if self.storage.alive.check_alive(module_type) is True:
            self.logger.error('{} still remains.'.format(module_type))
            sys.exit()

        self.storage.alive.set_any_process_state(module_type, 1)
        self.storage.pid.set_any_process_pid(module_type, os.getpid())

    def post_process(self) -> None:
        """Post-procedure after executed processes.

        Returns:
            None

        Raises:
            NotImplementedError: Causes when he inherited class does not
                implement.
        """
        raise NotImplementedError

    def start(self) -> None:
        """Start the all processes.

        Returns:
            None
        """
        self.pre_process()
        self.loop()
        self.post_process()

    def loop_pre_process(self) -> None:
        """Called before entering a main loop process.

        Returns:
            None
        """
        raise NotImplementedError

    def loop_post_process(self) -> None:
        """Called after exiting a main loop process.

        Returns:
            None

        Raises:
            NotImplementedError: Causes when the inherited class does not
                implement.
        """
        raise NotImplementedError

    def inner_loop_pre_process(self) -> None:
        """Called before executing a main loop process. This process is
            repeated every main loop.

        Returns:
            None

        Raises:
            NotImplementedError: Causes when the inherited class does not
                implement.
        """
        raise NotImplementedError

    def inner_loop_main_process(self) -> None:
        """A main loop process. This process is repeated every main loop.

        Returns:
            None

        Raises:
            NotImplementedError: Causes when the inherited class does not
                implement.
        """
        raise NotImplementedError

    def inner_loop_post_process(self) -> None:
        """Called after exiting a main loop process. This process is repeated
            every main loop.

        Returns:
            None

        Raises:
            NotImplementedError: Causes when the inherited class does not
                implement.
        """
        raise NotImplementedError

    def loop(self) -> None:
        """A loop process. This process is called after calling pre_process
            method, and is called before calling post_process.

        Returns:
            None
        """
        self.loop_pre_process()

        while True:
            if not self.inner_loop_pre_process():
                break

            if not self.inner_loop_main_process():
                break

            if not self.inner_loop_post_process():
                break

            self.wait()
            self.loop_count += 1

            if not self.check_error():
                break

        self.loop_post_process()

    def is_process_alive(self) -> bool:
        """Is processes(master, optimizer and scheduler) running or not.

        Returns:
            bool: Is processes running or not.
        """
        for pname in self.process_names:
            if self.storage.alive.check_alive(pname):
                if not is_process_running(self.storage.pid.get_any_process_pid(pname)):
                    return False
            else:
                return False
        return True

    def wait(self):
        time.sleep(self.sleep_time)

    def _serialize(self) -> None:
        """Serialize this module.

        Returns:
            None

        Raises:
            NotImplementedError: Causes when the inherited class does not
                implement.
        """
        raise NotImplementedError

    def _deserialize(self, trial_id: int) -> None:
        """Deserialize this module.

        Args:
            dict_objects(dict): A dictionary including serialized objects.

        Returns:
            None

        Raises:
            NotImplementedError: Causes when the inherited class does not
                implement.
        """
        raise NotImplementedError

    def set_native_random_seed(self) -> None:
        """ set any random seed.

        Args:
            None

        Returns:
            None
        """
        self.logger.debug('set native random seed: {}'.format(self.seed))
        random.seed(self.seed)

    def set_numpy_random_seed(self) -> None:
        """ set any random seed.

        Args:
            None

        Returns:
            None
        """
        self.logger.debug('set numpy random seed: {}'.format(self.seed))
        np.random.seed(seed=self.seed)

    def get_native_random_state(self) -> tuple:
        """ get random state.

        Args:
            None

        Returns:
            random.getstate (tuple)
        """
        return random.getstate()

    def set_native_random_state(self, state: tuple) -> None:
        """ get random state.

        Args:
            state (tuple): random state

        Returns:
            None
        """
        random.setstate(state)

    def get_numpy_random_state(self) -> tuple:
        """ get random state.

        Args:
            None

        Returns:
            numpy.random.get_state (tuple)
        """
        return np.random.get_state()

    def set_numpy_random_state(self, state: tuple) -> None:
        """ get random state.

        Args:
            state (tuple): random state

        Returns:
            None
        """
        np.random.set_state(state)

    def check_error(self) -> bool:
        """ Check to confirm if an error has occurred.

        Args:
            None

        Returns:
            True: no error | False: with error.
        """
        return True

    @property
    def current_max_trial_number(self) -> int:
        """ Get current trial number

        Args:
            None

        Returns:
            int: current trial number
        """
        return self.storage.current_max_trial_number()

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

    def stop(self) -> None:
        """ Stop optimization.

        Args:
            None

        Returns:
            None
        """
        self.storage.alive.init_alive()

    def get_zero_padding_any_trial_id(self, trial_id: int):
        return self.trial_id.zero_padding_any_trial_id(trial_id)
