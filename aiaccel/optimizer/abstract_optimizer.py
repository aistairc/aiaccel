import copy
from typing import Dict, List, Optional, Union

from aiaccel.module import AbstractModule
from aiaccel.parameter import load_parameter
from aiaccel.util.logger import str_to_logging_level
from aiaccel.util.trialid import TrialId


class AbstractOptimizer(AbstractModule):
    """An abstract class for Optimizer classes.

    Attributes:
        hp_total (int): A total number to generate hyper parameters.
        pool_size (int): A number to pool hyper parameters.
        params (HyperParameterConfiguration): Loaded hyper parameter
            configuration object.
        num_of_generated_parameter (int): A number of generated hyper paramters.
    """

    def __init__(self, options: dict) -> None:
        """Initial method of AbstractOptimizer.

        Args:
            config (str): A file name of a configuration.
        """
        self.options = options
        self.options['process_name'] = 'optimizer'
        super().__init__(self.options)

        self.set_logger(
            'root.optimizer',
            self.dict_log / self.config.optimizer_logfile.get(),
            str_to_logging_level(self.config.optimizer_file_log_level.get()),
            str_to_logging_level(self.config.optimizer_stream_log_level.get()),
            'Optimizer'
        )

        self.hp_ready = 0
        self.hp_running = 0
        self.hp_finished = 0
        self.num_of_generated_parameter = 0
        self.all_parameter_generated = False
        self.params = load_parameter(self.config.hyperparameters.get())
        self.trial_id = TrialId(str(self.config_path))

        self.storage.variable.register(
            process_name=self.options['process_name'],
            labels=['native_random_state', 'numpy_random_state', 'self_values', 'self_keys']
        )

    def register_new_parameters(self, params: List[dict]) -> None:
        """Create hyper parameter files.

        Args:
            params (List[dict]): A list of hyper parameter dictionaries.

        Returns:
            None

        Note:
            param = {
                'parameter_name': ...,
                'type': ...,
                'value': ...
            }
        """

        for param in params:
            self.register_ready(param)

    def register_ready(self, param: dict) -> str:
        """Create a hyper parameter file.

        Args:
            param (dict): A hyper parameter dictionary.

        Returns:
            str: An unique hyper parameter name.
        """

        self.trial_id.increment()

        param['trial_id'] = self.trial_id.get()
        self.storage.hp.set_any_trial_params(
            trial_id=param['trial_id'],
            params=param['parameters']
        )
        self.storage.trial.set_any_trial_state(
            trial_id=param['trial_id'],
            state='ready'
        )

        self._serialize(self.trial_id.integer)

        return param['trial_id']

    def generate_initial_parameter(self) -> Union[
        Dict[str, List[Dict[str, Union[str, Union[float, List[float]]]]]], None
    ]:
        """Generate a initial parameter.

        Returns:
            Union[Dict[str, List[Dict[str, Union[str, Union[float,
                List[float]]]]], None]: A created initial parameter. It returns
                None if any parameters are already created.
        """
        if self.num_of_generated_parameter == 0:
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
                self.num_of_generated_parameter += 1
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
        self.set_native_random_seed()
        self.set_numpy_random_seed()
        self.resume()

    def post_process(self) -> None:
        """Post-procedure after executed processes.

        Returns:
            None
        """
        self.logger.info('Optimizer delete alive file.')
        self.logger.info('Optimizer finished.')

    def loop_pre_process(self) -> None:
        """Called before entering a main loop process.

        Returns:
            None
        """
        return None

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
        if self.check_finished():
            return False

        return True

    def inner_loop_main_process(self) -> bool:
        """A main loop process. This process is repeated every main loop.

        Returns:
            bool: The process succeeds or not. The main loop exits if failed.
        """
        self.get_each_state_count()

        _max_pool_size = self.config.num_node.get()
        _max_trial_number = self.config.trial_number.get()

        n1 = _max_pool_size - self.hp_running - self.hp_ready
        n2 = _max_trial_number - self.hp_finished - self.hp_running - self.hp_ready
        pool_size = min(n1, n2)

        if self.hp_ready < _max_pool_size:
            self.logger.info(
                f'hp_ready: {self.hp_ready}, '
                f'hp_running: {self.hp_running}, '
                f'hp_finished: {self.hp_finished}, '
                f'total: {_max_trial_number}, '
                f'pool_size: {pool_size}'
            )
            self.generate_parameter(number=pool_size)
            if self.all_parameter_generated is True:
                self.logger.info("All parameter was generated.")
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

    def _serialize(self, trial_id: int) -> dict:
        """Serialize this module.
        Returns:
            dict: serialize data.
        """

        obj = self.__dict__.copy()
        del obj['storage']
        del obj['config']
        del obj['options']

        _values = list(obj.values())
        _keys = list(obj.keys())

        self.storage.variable.d['self_values'].set(trial_id, _values)
        self.storage.variable.d['self_keys'].set(trial_id, _keys)

        # random state
        self.storage.variable.d['native_random_state'].set(trial_id, self.get_native_random_state())
        self.storage.variable.d['numpy_random_state'].set(trial_id, self.get_numpy_random_state())

    def _deserialize(self, trial_id: int) -> None:
        """ Deserialize this module.
        Args:
            dict_objects(dict): A dictionary including serialized objects.
        Returns:
            None
        """
        _values = self.storage.variable.d['self_values'].get(trial_id)
        _keys = self.storage.variable.d['self_keys'].get(trial_id)
        self.__dict__.update(dict(zip(_keys, _values)))

        # random state
        self.set_native_random_state(self.storage.variable.d['native_random_state'].get(trial_id))
        self.set_numpy_random_state(self.storage.variable.d['numpy_random_state'].get(trial_id))

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

    def cast(self, param: dict) -> dict:
        _param = copy.deepcopy(param)
        param_type = _param['type']
        param_value = _param['value']

        try:
            if param_type.lower() == 'float':
                _param['value'] = float(param_value)
            if param_type.lower() == 'int':
                _param['value'] = int(param_value)
        except ValueError as e:
            raise ValueError(e)

        return _param
