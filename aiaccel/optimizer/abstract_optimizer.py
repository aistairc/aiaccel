from aiaccel.module import AbstractModule
from aiaccel.parameter import load_parameter
from aiaccel.util.filesystem import check_alive_file, create_yaml, file_delete
from aiaccel.util.logger import str_to_logging_level
from typing import Dict, List, Optional, Union
import aiaccel
import fasteners  # wd/
import logging
import time
from aiaccel.util.snapshot import SnapShot


class AbstractOptimizer(AbstractModule):
    """An abstract class for Optimizer classes.

    Attributes:
        hp_total (int): A total number to generate hyper parameters.
        pool_size (int): A number to pool hyper parameters.
        params (HyperParameterConfiguration): Loaded hyper parameter
            configuration object.
        generated_parameter (int): A number of generated hyper paramters.
    """

    def __init__(self, options: dict) -> None:
        """Initial method of AbstractOptimizer.

        Args:
            config (str): A file name of a configuration.
        """
        self.options = options
        super().__init__(self.options)

        self.set_logger(
            'root.optimizer',
            self.dict_log / self.config.optimizer_logfile.get(),
            str_to_logging_level(self.config.optimizer_file_log_level.get()),
            str_to_logging_level(self.config.optimizer_stream_log_level.get()),
            'Optimizer'
        )
        if self.options['dbg'] is True:
            self.config.silent_mode.set(False)
        else:
            self.remove_logger_handler()
            self.logfile = "optimizer.log"
            self.set_logger(
                'root.optimizer',
                self.dict_log / self.logfile,
                logging.DEBUG,
                logging.CRITICAL,
                'Optimizer'
            )

        self.exit_alive(self.alive_optimizer)
        self.hp_total = self.config.trial_number.get()
        self.hp_ready = 0
        self.hp_running = 0
        self.hp_finished = 0
        self.generated_parameter = 0
        self.sleep_time = self.config.sleep_time_optimizer.get()
        self.all_parameter_generated = False
        self.snapshot = SnapShot(self.ws, self.options['process_name'])
        self.params = load_parameter(self.config.hyperparameters.get())
        self.serialize_datas = {}

    def create_parameter_files(self, params: List[dict]) -> None:
        """Create hyper parameter files.

        Args:
            params (List[dict]): A list of hyper parameter dictionaries.

        Returns:
            None
        """
        for param in params:
            self.create_parameter_file(param)

    def create_parameter_file(self, param: dict) -> str:
        """Create a hyper parameter file.

        Args:
            param (dict): A hyper parameter dictionary.

        Returns:
            str: An unique hyper parameter name.
        """

        # wd/
        file_hp_count_fmt = '%0{}d'.format(self.config.name_length.get())
        count_path = self.dict_hp / aiaccel.file_hp_count
        lock_path = self.dict_hp / aiaccel.file_hp_count_lock
        lock = fasteners.InterProcessLock(str(lock_path))
        if lock.acquire(timeout=aiaccel.file_hp_count_lock_timeout):
            number = 0
            if count_path.exists():
                number = int(count_path.read_text())
                number += 1
            count_path.write_text('%d' % number)
            lock.release()
            name = file_hp_count_fmt % number
        else:
            self.logger.error('lock timeout {}'.format(lock_path))
            # In the original source, when generate_random_name() returns
            # I didn't make it an error if it returned None.
            name = None
        filename = '{}.hp'.format(name)

        param['hashname'] = name
        create_yaml(
            (self.ws / aiaccel.dict_hp_ready / filename),
            param,
            self.dict_lock
        )
        self.hp_ready += 1

        return name

    def generate_initial_parameter(self) ->\
            Union[
                Dict[str,
                     List[Dict[str, Union[str, Union[float, List[float]]]]]],
                None]:
        """Generate a initial parameter.

        Returns:
            Union[Dict[str, List[Dict[str, Union[str, Union[float,
                List[float]]]]], None]: A created initial parameter. It returns
                None if any parameters are already created.
        """
        if self.generated_parameter == 0:
            sample = self.params.sample(initial=True)
            new_params = []

            for s in sample:
                new_param = {
                    'parameter_name': s['name'],
                    'type': s['type'],
                    'value': s['value']
                }
                new_params.append(new_param)

            if len(new_params) == len(self.params.get_parameter_list()):
                self.generated_parameter += 1
                return {'parameters': new_params}

        return None

    def generate_parameter(self, number: Optional[int] = 1) -> None:
        """Generate parameters.

        Args:
            number (Optional[int]): A number of generating parameters.

        Returns:
            None

        Raises:
            NotImplementedError: Causes when the inherited class does not
                implement.
        """
        raise NotImplementedError

    def pre_process(self) -> None:
        """Pre-procedure before executing processes.

        Returns:
            None
        """
        super().pre_process()

    def post_process(self) -> None:
        """Post-procedure after executed processes.

        Returns:
            None
        """
        file_delete(self.alive_optimizer, self.dict_lock)
        self.logger.info('Optimizer delete alive file.')
        self.logger.info('Optimizer finished.')

    def loop_pre_process(self) -> None:
        """Called before entering a main loop process.

        Returns:
            None
        """
        while not self.check_work_directory():
            time.sleep(self.sleep_time)

    def loop_post_process(self) -> None:
        """Called after exiting a main loop process.

        Returns:
            None
        """
        return None

    def inner_loop_pre_process(self) -> bool:
        """Called before executing a main loop process. This process is
            repeated every main loop.

        Returns:
            bool: The process succeeds or not. The main loop exits if failed.
        """
        if not check_alive_file(self.alive_optimizer, self.dict_lock):
            self.logger.info('The alive file of optimizer is deleted')
            return False

        if self.check_finished():
            return False

        return True

    def inner_loop_main_process(self) -> bool:
        """A main loop process. This process is repeated every main loop.

        Returns:
            bool: The process succeeds or not. The main loop exits if failed.
        """
        self.get_dict_state()

        _max_pool_size = self.config.num_node.get()
        n1 = _max_pool_size - self.hp_running - self.hp_ready
        n2 = self.hp_total - self.hp_finished - self.hp_running - self.hp_ready
        pool_size = min(n1, n2)

        if self.hp_ready < _max_pool_size:
            self.logger.info(
                'hp_ready: {}, hp_running: {}, hp_finished: {}, '
                'total: {}, pool_size: {}'
                .format(
                    self.hp_ready,
                    self.hp_running,
                    self.hp_finished,
                    self.hp_total,
                    pool_size
                )
            )
            self.generate_parameter(number=pool_size)
            if self.all_parameter_generated is True:
                print("All parameter was generated.")
                return False

        return True

    def inner_loop_post_process(self) -> bool:
        """Called after exiting a main loop process. This process is repeated
            every main loop.

        Returns:
            bool: The process succeeds or not. The main loop exits if failed.
        """
        self.print_dict_state()
        return True

    def _serialize(self) -> dict:
        """Serialize this module.

        Returns:
            None
        """
        if self.options['nosave'] is True:
            pass
        else:
            self.snapshot.save(
                self.curr_trial_number,
                self.loop_count,
                self.get_native_random_state(),
                self.get_numpy_random_state(),
                self.serialize_datas
            )
        return self.serialize_datas

    def _deserialize(self, dict_objects: dict) -> None:
        """Deserialize this module.

        Args:
            dict_objects(dict): A dictionary including serialized objects.

        Returns:
            None
        """

        self.generated_parameter = dict_objects['generated_parameter']
        loop_counts = (
            self.snapshot.get_inner_loop_counter(self.options['resume'])
        )
        if loop_counts is None:
            return

        self.loop_count = loop_counts['optimizer']
        print(
            "({})set inner loop count: {}"
            .format('optimizer', self.loop_count)
        )
