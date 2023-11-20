from __future__ import annotations

import copy
from typing import Any

from numpy import isnan
from omegaconf.dictconfig import DictConfig

from aiaccel.config import is_multi_objective
from aiaccel.converted_parameter import ConvertedIntParameter
from aiaccel.module import AbstractModule
from aiaccel.parameter import HyperParameterConfiguration, IntParameter, Parameter
from aiaccel.util import TrialId


class AbstractOptimizer(AbstractModule):
    """An abstract class for Optimizer classes.

    Args:
        config (DictConfig): A DictConfig object which contains optimization
            settings specified by the configuration file and the command line
            options.

    Attributes:
        hp_ready (int): The number of ready parameters which are registered
            with the storage and are not tried yet in the user program. The
            state label in the storage is "ready".
        hp_running (int): The number of parameters which the user program is
            running with. The state label in the storage is "running".
        hp_finished (int): The number of finished parameters which objective
            values obtained from the user program executions with are
            registered with the storage. The state label in the storage is
            "finished".
        num_of_generated_parameter (int): The number of generated paramters.
        all_parameters_generated (bool): Whether all parameters are generated.
            True if all parameters are generated.
        params (HyperParameterConfiguration): A loaded parameter configuration
            object.
        trial_id (TrialId): A TrialId object.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config, "optimizer")
        self.set_logger(
            logger_name="root.optimizer",
            logfile=self.workspace.log / "optimizer.log",
            file_level=self.config.generic.logging_level,
            stream_level=self.config.generic.logging_level,
        )

        self.trial_number = self.config.optimize.trial_number
        self.hp_ready = 0
        self.hp_running = 0
        self.hp_finished = 0
        self.num_of_generated_parameter = 0
        self.params = HyperParameterConfiguration(self.config.optimize.parameters)
        self.trial_id = TrialId(self.config)
        self.all_parameters_generated = False

    def get_trial_id(self) -> int:
        """Get the current trial ID.

        Returns:
            int: The current trial ID.
        """
        return self.trial_id.integer

    def is_all_parameters_generated(self) -> bool:
        """Check if all parameters are generated.

        Returns:
            bool: True if all parameters are generated.
        """
        return (self.num_of_generated_parameter >= self.trial_number) or self.all_parameters_generated

    def register_new_parameters(self, params: list[dict[str, float | int | str]], state: str = "ready") -> None:
        """Create hyper parameter files.

        Args:
            params (list[dict[str, float | int | str]]): A list of hyper
                parameter dictionaries.

        Returns:
            None

        Note:
            ::

                param = {
                    'name': ...,
                    'type': ...,
                    'value': ...
                }

        """
        self.storage.hp.set_any_trial_params(trial_id=self.trial_id.get(), params=params)
        self.storage.trial.set_any_trial_state(trial_id=self.trial_id.get(), state=state)
        self.num_of_generated_parameter += 1
        self.logger.debug(f"generated parameters: {params}")

    def generate_initial_parameter(self) -> Any:
        """Generate a list of initial parameters.

        Returns:
            list[dict[str, float | int | str]]: A created list of initial
            parameters.
        """
        sample = self.params.sample(self._rng, initial=True)
        new_params = []

        for s in sample:
            new_param = {"parameter_name": s["name"], "type": s["type"], "value": s["value"]}
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

    def generate_new_parameter(self) -> list[dict[str, float | int | str]] | None:
        """Generate a list of parameters.

        Returns:
            list[dict[str, float | int | str]] | None: A created list of
            parameters.
        """
        if self.num_of_generated_parameter == 0:
            new_params = self.generate_initial_parameter()
        else:
            new_params = self.generate_parameter()
        return new_params

    def pre_process(self) -> None:
        """Pre-procedure before executing processes.

        Returns:
            None
        """
        self.write_random_seed_to_debug_log()
        self.resume()

    def post_process(self) -> None:
        """Post-procedure after executed processes.

        Returns:
            None
        """
        self.logger.info("Optimizer delete alive file.")
        self.logger.info("Optimizer finished.")

    def convert_type_by_config(
        self, temp_new_params: list[dict[str, float | int | str]]
    ) -> list[dict[str, float | int | str]]:
        """Convert the type of parameters by the configuration file.

        Args:
            new_params (list[dict[str, float | int | str]]): A list of
                parameters.

        Returns:
            list[dict[str, float | int | str]]: A list of converted parameters.
        """
        new_params = copy.deepcopy(temp_new_params)
        config_params: dict[str, Parameter] = self.params.get_parameter_dict()
        for new_param in new_params:
            name = str(new_param["parameter_name"])
            if isinstance(config_params[name], IntParameter) or isinstance(config_params[name], ConvertedIntParameter):
                if isnan(new_param["value"]):
                    continue
                new_param["value"] = int(new_param["value"])
        return new_params

    def run_optimizer(self) -> None:
        if new_params := self.generate_new_parameter():
            self.register_new_parameters(self.convert_type_by_config(new_params))
            self.trial_id.increment()
            self.serialize(self.trial_id.integer)

    def resume(self) -> None:
        """When in resume mode, load the previous optimization data in advance.

        Args:
            None

        Returns:
            None
        """
        self.logger.info(f"Resume mode: {self.config.resume}")
        self.trial_id.initial(num=self.config.resume)
        super().deserialize(self.config.resume)
        self.trial_number = self.config.optimize.trial_number

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

    def is_error_free(self) -> bool:
        """Check if all trials are error free.

        Returns:
            bool: True if all trials are error free.
        """
        error_trial_ids = self.storage.error.get_error_trial_id()
        failed_trial_ids = self.storage.error.get_failed_exitcode_trial_id()

        if self.config.generic.is_ignore_warning:
            if len(failed_trial_ids) == 0:
                return True
        else:
            if len(failed_trial_ids) == 0 and len(error_trial_ids) == 0:
                return True

        for trial_id in error_trial_ids:
            error_message = self.storage.error.get_any_trial_error(trial_id=trial_id)
            self.logger.error(error_message)

        return False

    def finalize_operation(self) -> None:
        """Finalize the operation.

        Args:
            None

        Returns:
            None
        """
        ...
