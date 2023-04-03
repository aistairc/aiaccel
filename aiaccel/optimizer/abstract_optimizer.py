from __future__ import annotations

import copy
from typing import Any

from numpy import str_

from aiaccel.config import is_multi_objective
from aiaccel.module import AbstractModule
from aiaccel.parameter import load_parameter
from aiaccel.util import TrialId, str_to_logging_level


class AbstractOptimizer(AbstractModule):
    """An abstract class for Optimizer classes.

    Args:
        options (dict[str, str | int | bool]): A dictionary containing
            command line options.

    Attributes:
        options (dict[str, str | int | bool]): A dictionary containing
            command line options.
        hp_ready (int): A ready number of hyperparameters.
        hp_running (int): A running number of hyperprameters.
        hp_finished (int): A finished number of hyperparameters.
        num_of_generated_parameter (int): A number of generated hyperparamters.
        all_parameters_generated (bool): Whether all parameters are generated.
            True if all parameters are generated.
        params (HyperParameterConfiguration): Loaded hyper parameter
            configuration object.
        trial_id (TrialId): TrialId object.
    """

    def __init__(self, options: dict[str, str | int | bool]) -> None:
        self.options = options
        self.options['process_name'] = 'optimizer'
        super().__init__(self.options)

        self.set_logger(
            'root.optimizer',
            self.workspace.log / self.config.optimizer_logfile.get(),
            str_to_logging_level(self.config.optimizer_file_log_level.get()),
            str_to_logging_level(self.config.optimizer_stream_log_level.get()),
            'Optimizer'
        )

        self.trial_number = self.config.trial_number.get()
        self.hp_ready = 0
        self.hp_running = 0
        self.hp_finished = 0
        self.num_of_generated_parameter = 0
        self.params = load_parameter(self.config.hyperparameters.get())
        self.trial_id = TrialId(str(self.config_path))
        self.all_parameters_generated = False

    def all_parameters_processed(self) -> bool:
        """Checks whether any unprocessed parameters are left.

        This method is beneficial for the case that the maximum number of
        parameter generation is limited by algorithm (e.g. grid search).
        To make this method effective, the algorithm with the parameter
        generation limit should turn `all_parameters_generated` True when all
        of available parameters are generated.

        Returns:
            bool: True if all parameters are generated and are processed.
        """
        return self.hp_ready == 0 and self.hp_running == 0 and self.all_parameters_generated

    def all_parameters_registered(self) -> bool:
        """Checks whether all parameters that can be generated with the given
        number of trials are registered.

        This method does not check whether the registered parameters have been
        processed.

        Returns:
            bool: True if all parameters are registerd.
        """
        return self.trial_number - self.hp_finished - self.hp_ready - self.hp_running == 0

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
        sample = self.params.sample(self._rng, initial=True)
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
        max_pool_size = self.config.num_node.get()
        hp_running = self.storage.get_num_running()
        hp_ready = self.storage.get_num_ready()
        available_pool_size = max_pool_size - hp_running - hp_ready
        return available_pool_size

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
        self.update_each_state_count()

        if self.check_finished():
            return False

        if self.all_parameters_processed():
            return False

        if self.all_parameters_registered():
            return True

        pool_size = self.get_pool_size()
        if pool_size == 0:
            return True

        self.logger.info(
            f'hp_ready: {self.hp_ready}, '
            f'hp_running: {self.hp_running}, '
            f'hp_finished: {self.hp_finished}, '
            f'total: {self.config.trial_number.get()}, '
            f'pool_size: {pool_size}'
        )

        if new_params := self.generate_new_parameter():
            self.register_new_parameters(new_params)
            self.trial_id.increment()
            self._serialize(self.trial_id.integer)
            return True

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
            self.options['resume'] is not None and
            self.options['resume'] > 0
        ):
            self.storage.rollback_to_ready(self.options['resume'])
            self.storage.delete_trial_data_after_this(self.options['resume'])
            self.trial_id.initial(num=self.options['resume'])
            self._deserialize(self.options['resume'])
            self.trial_number = self.config.trial_number.get()

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

    def get_any_trial_objective(self, trial_id: int) -> Any:
        """Get any trial result.

            if the objective is multi-objective, return the list of objective.

        Args:
            trial_id (int): Trial ID.

        Returns:
            Any: Any trial result.
        """

        objective = self.storage.result.get_any_trial_objective(trial_id)
        if objective is None:
            return None

        if is_multi_objective(self.config):
            return objective
        else:
            return objective[0]
