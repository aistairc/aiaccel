from __future__ import annotations

import copy
from typing import Any

from numpy import str_
from omegaconf.dictconfig import DictConfig

from aiaccel.module import AbstractModule
from aiaccel.parameter import HyperParameterConfiguration
from aiaccel.util.logger import str_to_logging_level
from aiaccel.util.trialid import TrialId


class AbstractOptimizer(AbstractModule):
    """An abstract class for Optimizer classes.

    Args:
        options (dict[str, str | int | bool]): A dictionary containing
            command line options.

    Attributes:
        options (dict[str, str | int | bool]): A dictionary containing
            command line options.
        hp_ready (int): A ready number of hyper parameters.
        hp_running (int): A running number of hyper prameters.
        hp_finished (int): A finished number of hyper parameters.
        num_of_generated_parameter (int): A number of generated hyper
            paramters.
        all_parameter_generated (bool): A boolean indicating if all parameters
            are generated or not.
        params (HyperParameterConfiguration): Loaded hyper parameter
            configuration object.
        trial_id (TrialId): TrialId object.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config, 'optimizer')
        self.set_logger(
            'root.optimizer',
            self.dict_log / self.config.logger.file.optimizer,
            str_to_logging_level(self.config.logger.log_level.optimizer),
            str_to_logging_level(self.config.logger.stream_level.optimizer),
            'Optimizer'
        )

        self.hp_ready = 0
        self.hp_running = 0
        self.hp_finished = 0
        self.num_of_generated_parameter = 0
        self.all_parameter_generated = False
        self.params = HyperParameterConfiguration(self.config.optimize.parameters)
        self.trial_id = TrialId(self.config.config_path)

    def register_new_parameters(
        self,
        params: list[dict[str, float | int | str]]
    ) -> None:
        """Create hyper parameter files.

        Args:
            params (list[dict[str, float | int | str]]): A list of hyper
                parameter dictionaries.

        Returns:
            None

        Note:
            ::

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

    def generate_initial_parameter(
        self
    ) -> Any:
        """Generate a list of initial parameters.

        Returns:
            list[dict[str, float | int | str]]: A created list of initial
            parameters.
        """
        sample = self.params.sample(initial=True, rng=self._rng)
        new_params = []

        for s in sample:
            new_param = {
                'parameter_name': s['name'],
                'type': s['type'],
                'value': s['value']
            }
            new_params.append(new_param)

        return new_params

    def generate_parameter(self) -> Any:
        """Generate a list of parameters.

        Raises:
            NotImplementedError: Causes when the inherited class does not
                implement.

        Returns:
            list[dict[str, float | int | str]] | None: A created list of
            parameters.
        """
        raise NotImplementedError

    def get_pool_size(self) -> int:
        """Returns pool size.

        Returns:
            int: Pool size.
        """
        hp_ready = self.storage.get_num_ready()
        hp_running = self.storage.get_num_running()
        hp_finished = self.storage.get_num_finished()

        max_pool_size = self.config.resource.num_node
        max_trial_number = self.config.optimize.trial_number

        n1 = max_pool_size - hp_running - hp_ready
        n2 = max_trial_number - hp_finished - hp_running - hp_ready
        pool_size = min(n1, n2)

        if (pool_size <= 0 or hp_ready >= max_pool_size):
            return 0

        return pool_size

    def generate_new_parameter(self) -> list[dict[str, float | int | str]] | None:
        """Generate a list of parameters.

        Returns:
            list[dict[str, float | int | str]] | None: A created list of
            parameters.
        """
        if self.num_of_generated_parameter == 0:
            new_params = self.cast(self.generate_initial_parameter())
        else:
            new_params = self.cast(self.generate_parameter())

        return new_params

    def pre_process(self) -> None:
        """Pre-procedure before executing processes.

        Returns:
            None
        """
        self.create_numpy_random_generator()
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

        pool_size = self.get_pool_size()
        if pool_size <= 0:
            return True

        self.logger.info(
            f'hp_ready: {self.hp_ready}, '
            f'hp_running: {self.hp_running}, '
            f'hp_finished: {self.hp_finished}, '
            f'total: {self.config.optimize.trial_number}, '
            f'pool_size: {pool_size}'
        )

        for _ in range(pool_size):
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

    def resume(self) -> None:
        """ When in resume mode, load the previous optimization data in advance.

        Args:
            None

        Returns:
            None
        """
        if (
            self.config.resume is not None and
            self.config.resume > 0
        ):
            self.storage.rollback_to_ready(self.config.resume)
            self.storage.delete_trial_data_after_this(self.config.resume)
            self.trial_id.initial(num=self.config.resume)
            self._deserialize(self.config.resume)

    def cast(self, params: list[dict[str, Any]]) -> list[Any] | None:
        """Casts types of parameter values to appropriate tepes.

        Args:
            params (list | None): list of parameters.

        Raises:
            ValueError: Occurs if any of parameter value could not be casted.

        Returns:
            list | None: A list of parameters with casted values. None if given
            `params` is None.
        """
        if params is None or len(params) == 0:
            return params

        casted_params = []

        for param in params:
            _param = copy.deepcopy(param)
            param_type = _param['type']
            param_value = _param['value']

            # None: str to NoneType
            if type(_param['value']) in [str, str_]:
                if _param['value'].lower() == 'none':
                    _param['value'] = None
                    _param['type'] = str(type(None))

            try:
                if (
                    param_type.lower() == 'categorical' or
                    param_type.lower() == 'ordinal'
                ):
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
        """Checks errors.

        Returns:
            bool: True if there is no error.
        """
        error_trial_ids = self.storage.error.get_error_trial_id()
        if len(error_trial_ids) == 0:
            return True

        for trial_id in error_trial_ids:
            error_message = self.storage.error.get_any_trial_error(
                trial_id=trial_id
            )
            self.logger.error(error_message)

        return False
