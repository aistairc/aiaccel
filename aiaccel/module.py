import aiaccel
from aiaccel.config import Config
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

        # logger
        self.logger = None
        self.fh = None
        self.ch = None
        self.ch_formatter = None
        self.ch_formatter = None

        self.loop_count = 0
        self.hp_total = self.config.trial_number.get()
        self.hp_ready = 0
        self.hp_running = 0
        self.hp_finished = 0

        self.seed = self.config.randseed.get()
        self.storage = Storage(
            self.ws,
            fsmode=options['fs'],
            config_path=self.config.config_path
        )
        self.trial_id = TrialId(self.options['config'])
        self.serialize_datas = {}
        self.deserialize_datas = {}

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
        if aiaccel.class_optimizer in self.__class__.__name__:
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

    def remove_logger_handler(self):
        self.logger = None

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
        self.logger.debug(f'set native random seed: {self.seed}')
        random.seed(self.seed)

    def set_numpy_random_seed(self) -> None:
        """ set any random seed.

        Args:
            None

        Returns:
            None
        """
        self.logger.debug(f'set numpy random seed: {self.seed}')
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

    def get_zero_padding_any_trial_id(self, trial_id: int):
        return self.trial_id.zero_padding_any_trial_id(trial_id)
