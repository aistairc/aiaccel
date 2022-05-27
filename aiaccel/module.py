import aiaccel
import fasteners
import logging
import os
import threading
import time
import sys
from aiaccel.config import Config
from aiaccel.util.filesystem import check_alive_file
from aiaccel.util.filesystem import create_yaml
from aiaccel.util.filesystem import get_file_hp_finished
from aiaccel.util.filesystem import get_file_hp_ready
from aiaccel.util.filesystem import get_file_hp_running
from aiaccel.util.filesystem import interprocess_lock_file
from aiaccel.util.filesystem import load_yaml
from aiaccel.util.filesystem import make_directories
from aiaccel.util.filesystem import make_directory
from aiaccel.util.process import is_process_running
from multiprocessing import Barrier
from pathlib import Path
from typing import Tuple
import numpy as np
import random
from aiaccel.util.snapshot import SnapShot


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
        barrier (multiprocessing.Barrier): A barrier to synchronize processes.
        config (ConfileWrapper): A config object.
        dict_alive (Path): A path to alive directory.
        dict_hp (Path): A path to hp directory.
        dict_lock (Path): A path to lock directory.
        dict_log (Path): A path to log directory.
        dict_output (Path): A path to output directory.
        dict_runner (Path): A path to runner directory.
        dict_state (Path): A path to state directory.
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

        self.logger = None
        self.ch = None
        self.ch_formatter = None
        self.ch_formatter = None
        self.loop_count = 0
        self.hp_ready = 0
        self.hp_running = 0
        self.hp_finished = 0
        self.dict_alive = self.ws / aiaccel.dict_alive
        self.dict_hp = self.ws / aiaccel.dict_hp
        self.dict_lock = self.ws / aiaccel.dict_lock
        self.dict_log = self.ws / aiaccel.dict_log
        self.dict_output = self.ws / aiaccel.dict_output
        self.dict_result = self.ws / aiaccel.dict_result
        self.dict_runner = self.ws / aiaccel.dict_runner
        self.dict_state = self.ws / aiaccel.dict_state
        self.dict_verification = self.ws / aiaccel.dict_verification
        self.alive_master = self.dict_alive / aiaccel.alive_master
        self.alive_optimizer = self.dict_alive / aiaccel.alive_optimizer
        self.alive_scheduler = self.dict_alive / aiaccel.alive_scheduler
        self.dict_hp_ready = self.ws / aiaccel.dict_hp_ready
        self.dict_hp_running = self.ws / aiaccel.dict_hp_running
        self.dict_hp_finished = self.ws / aiaccel.dict_hp_finished
        self.sleep_time = 1.0
        self.barrier = None
        self.seed = self.config.randseed.get()
        self.snapshot = SnapShot(self.ws, self.options['process_name'])
        self.barrier_timeout = (self.config.batch_job_timeout.get() * self.config.job_retry.get())

    def make_work_directory(self) -> None:
        """Create a work directory.

        Returns:
            None

        Raises:
            NotADirectoryError: It raises if a workspace argument (self.ws) is
            not a directory.
        """
        if not self.ws.is_dir():
            self.logger.error(
                'Invalid work path: {}, is set in config.'.format(self.ws)
            )
            raise NotADirectoryError(
                'Invalid work path: {}, is set in config.'.format(self.ws)
            )

        for d in [self.dict_lock]:
            if not d.is_dir():
                if d.exists():
                    os.remove(d)
                make_directory(d)

        make_directories(
            [
                self.dict_alive,
                self.dict_hp,
                self.dict_log,
                self.dict_output,
                self.dict_result,
                self.dict_runner,
                self.dict_state,
                self.dict_verification
            ],
            self.dict_lock
        )

        make_directories(
            [
                self.dict_hp_ready, self.dict_hp_running, self.dict_hp_finished
            ],
            self.dict_lock
        )

    def check_work_directory(self) -> bool:
        """Check required directories exist or not.

        Returns:
            bool: All required directories exist or not.
        """

        dirs = [
            self.dict_alive,
            self.dict_hp,
            self.dict_lock,
            self.dict_log,
            self.dict_output,
            self.dict_result,
            self.dict_runner,
            self.dict_verification,
            self.dict_hp_ready,
            self.dict_hp_running,
            self.dict_hp_finished
        ]

        for d in dirs:
            with fasteners.InterProcessLock(
                    interprocess_lock_file(d, self.dict_lock)):
                if d.is_dir():
                    continue
                else:
                    return False

        return True

    def get_dict_state(self) -> None:
        """Updates the number of files in hp(hyper parameter) directories.

        Returns:
            None
        """
        self.hp_ready = len(get_file_hp_ready(self.ws, self.dict_lock))
        self.hp_running = len(get_file_hp_running(self.ws, self.dict_lock))
        self.hp_finished = len(get_file_hp_finished(self.ws, self.dict_lock))

    def get_module_type_alive_file(self) -> Tuple[str, Path]:
        """Get this module type and a path to alive file.

        Returns:
            Tuple[str, Path]: This module type and a path to alive file.
        """
        with fasteners.InterProcessLock(
            interprocess_lock_file(self.dict_alive, self.dict_lock)
        ):
            if aiaccel.class_master in self.__class__.__name__:
                module_type = aiaccel.module_type_master
                alive_file = self.alive_master
            elif aiaccel.class_optimizer in self.__class__.__name__:
                module_type = aiaccel.module_type_optimizer
                alive_file = self.alive_optimizer
            elif aiaccel.class_scheduler in self.__class__.__name__:
                module_type = aiaccel.module_type_scheduler
                alive_file = self.alive_scheduler
            else:
                module_type = None
                alive_file = None
                self.logger.error(
                    'Unknown type of module: {}'
                    .format(self.__class__.__name__)
                )

        return module_type, alive_file

    def check_finished(self) -> bool:
        """Check whether all optimization finished or not.

        Returns:
            bool: All optimization finished or not.
        """
        with fasteners.InterProcessLock(
            interprocess_lock_file(self.dict_hp, self.dict_lock)
        ):
            files = get_file_hp_finished(self.ws)

        self.hp_finished = len(files)

        if self.hp_finished >= self.config.trial_number.get():
            return True

        return False

    def exit_alive(self, filename: Path) -> None:
        """Exit the execution if alive files exist.

        Args:
            filename (Path): A path to an alive file.

        Returns:
            None
        """
        with fasteners.InterProcessLock(
            interprocess_lock_file(self.dict_alive, self.dict_lock)
        ):
            if check_alive_file(filename, self.dict_lock):
                self.logger.info('Alive file exist: {}'.format(filename))
                self.logger.info(
                    'Please confirm the previous execution is '
                    'finished, and delete the alive file.'
                )
                sys.exit()

    def print_dict_state(self) -> None:
        """Print hp(hyperparameter) directory states.

        Returns:
            None
        """
        self.logger.info('{}/{} finished, ready: {}, running: {}'.format(
            self.hp_finished,
            self.config.trial_number.get(),
            self.hp_ready,
            self.hp_running))

    def set_barrier(self, barrier: Barrier) -> None:
        """Set a multiprocessing barrier.

        Args:
            barrier (multiprocessing.Barrier): A barrier object.

        Returns:
            None
        """
        self.barrier = barrier

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
        module_type, alive_file = self.get_module_type_alive_file()

        if check_alive_file(alive_file, self.dict_lock):
            self.logger.error('Alive file still remains: {}.'.format(
                alive_file))
            sys.exit()

        yml = {
            aiaccel.key_pid: os.getpid(),
            aiaccel.key_path: str(self.ws),
            aiaccel.key_module_type: module_type
        }
        create_yaml(alive_file, yml, self.dict_lock)
        self.set_native_random_seed()
        self.set_numpy_random_seed()
        self.resume()

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
        while not self.check_work_directory():
            time.sleep(self.config.sleep_time_master.get())

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

            if not self.is_barrier():
                break

            if not self.check_error():
                break
            self._serialize()

            if not self.is_barrier():
                break

        self.loop_post_process()

    def is_barrier(self) -> bool:
        """Is barrier waiting working well or not.

        Returns:
            bool: It returns false if the barrier object is not set or the
            processes are not running. Otherwise, it returns true.
        """

        start_time = time.time()
        check_cycle_time = 60
        while self.is_process_alive() and (time.time() - start_time) < self.barrier_timeout:
            try:
                self.barrier.wait(check_cycle_time)
                return True
            except threading.BrokenBarrierError:
                self.wait()
        return False

    def is_process_alive(self) -> bool:
        """Is processes(master, optimizer and scheduler) running or not.

        Returns:
            bool: Is processes running or not.
        """
        alive_files = [
            self.alive_master,
            self.alive_scheduler,
            self.alive_optimizer
        ]
        for alive_file in alive_files:
            if alive_file.exists():
                obj = load_yaml(alive_file, self.dict_lock)
                if not is_process_running(obj['pid']):
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

    def _deserialize(self, dict_objects: dict) -> None:
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
        self.logger.debug(
            'set native random seed: {}'.format(self.seed)
        )
        random.seed(self.seed)

    def set_numpy_random_seed(self) -> None:
        """ set any random seed.

        Args:
            None

        Returns:
            None
        """
        self.logger.debug(
            'set numpy random seed: {}'.format(self.seed)
        )
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
        self.logger.debug(
            'set native random state: {}'.format(state)
        )
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
        self.logger.debug(
            'set numpy random state: {}'.format(state)
        )
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
    def curr_trial_number(self) -> int:
        """ Get current trial number

        Args:
            None

        Returns:
            int: current trial number
        """
        return len(list(self.dict_hp_finished.glob("*.hp")))

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
            self.snapshot.load(self.options['resume'])
            self.set_native_random_state(self.snapshot.random_state_native)
            self.set_numpy_random_state(self.snapshot.random_state_numpy)
            self._deserialize(self.snapshot.process_memory_objects)

    def stop(self) -> None:
        """ Stop optimization.

        Args:
            None

        Returns:
            None
        """
        alive_files = [
            self.alive_master,
            self.alive_scheduler,
            self.alive_optimizer
        ]
        for alive_file in alive_files:
            if alive_file.exists():
                alive_file.unlink()
