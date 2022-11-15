import copy
from typing import Dict, List, Union

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
        self.storage.hp.set_any_trial_params(
            trial_id=self.trial_id.get(),
            params=params
        )

        self.storage.trial.set_any_trial_state(
            trial_id=self.trial_id.get(),
            state='ready'
        )

        self.num_of_generated_parameter += 1

    def generate_initial_parameter(self) -> Union[
        Dict[str, List[Dict[str, Union[str, Union[float, List[float]]]]]], None
    ]:
        """Generate a initial parameter.

        Returns:
            Union[Dict[str, List[Dict[str, Union[str, Union[float,
                List[float]]]]], None]: A created initial parameter. It returns
                None if any parameters are already created.
        """

        sample = self.params.sample(initial=True)
        new_params = []

        for s in sample:
            new_param = {
                'parameter_name': s['name'],
                'type': s['type'],
                'value': s['value']
            }
            new_params.append(new_param)

        return new_params

    def generate_parameter(self) -> list:
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

    def generate_new_parameter(self) -> list:
        if self.num_of_generated_parameter == 0:
            new_params = self.cast(self.generate_initial_parameter())
        else:
            new_params = self.cast(self.generate_parameter())

        return new_params

    def get_pool_size(self) -> int:

        hp_ready = self.storage.get_num_ready()
        hp_running = self.storage.get_num_running()
        hp_finished = self.storage.get_num_finished()

        max_pool_size = self.config.num_node.get()
        max_trial_number = self.config.trial_number.get()

        n1 = max_pool_size - hp_running - hp_ready
        n2 = max_trial_number - hp_finished - hp_running - hp_ready
        pool_size = min(n1, n2)

        return pool_size


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

    def inner_loop_main_process(self) -> bool:
        """A main loop process. This process is repeated every main loop.

        Returns:
            bool: The process succeeds or not. The main loop exits if failed.
        """

        if self.check_finished():
            return False

        self.get_each_state_count()

        _max_pool_size = self.config.num_node.get()
        _max_trial_number = self.config.trial_number.get()

        n1 = _max_pool_size - self.hp_running - self.hp_ready
        n2 = _max_trial_number - self.hp_finished - self.hp_running - self.hp_ready
        pool_size = min(n1, n2)

        if (pool_size <= 0 or self.hp_ready >= _max_pool_size):
            return True

        self.logger.info(
            f'hp_ready: {self.hp_ready}, '
            f'hp_running: {self.hp_running}, '
            f'hp_finished: {self.hp_finished}, '
            f'total: {_max_trial_number}, '
            f'pool_size: {pool_size}'
        )

        new_params = self.generate_new_parameter()
        if new_params is not None and len(new_params) > 0:
            self.register_new_parameters(new_params)

            self.trial_id.increment()
            self._serialize(self.trial_id.integer)

            if self.all_parameter_generated is True:
                self.logger.info("All parameter was generated.")
                return False

        self.print_dict_state()

        return True

    def _serialize(self, trial_id: int) -> None:
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
            self.storage.rollback_to_ready(self.options['resume'])
            self.storage.delete_trial_data_after_this(self.options['resume'])
            self.trial_id.initial(num=self.options['resume'])
            self._deserialize(self.options['resume'])

    def cast(self, params: Union[None, list]) -> Union[None, list]:
        if params is None or len(params) == 0:
            return params

        casted_params = []

        for param in params:
            _param = copy.deepcopy(param)
            param_type = _param['type']
            param_value = _param['value']

            try:
                if param_type.lower() == 'categorical' or param_type.lower() == 'ordinal':
                    casted_params.append(_param)
                    continue

                if type(_param['value']) == eval(param_type.lower()):
                    casted_params.append(_param)
                    continue

                if param_type.lower() == 'float':
                    _param['value'] = float(param_value)
                if param_type.lower() == 'int':
                    _param['value'] = int(param_value)
                casted_params.append(_param)

            except ValueError as e:
                raise ValueError(e)

        return casted_params

    def check_error(self) -> bool:
        error_trial_ids = self.storage.error.get_error_trial_id()
        if len(error_trial_ids) == 0:
            return True

        for trial_id in error_trial_ids:
            error_message = self.storage.error.get_any_trial_error(trial_id=trial_id)
            self.logger.error(error_message)

        return False
