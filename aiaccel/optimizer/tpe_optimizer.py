from __future__ import annotations

from typing import Any

import optuna

import aiaccel.parameter
from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer


class TPESamplerWrapper(optuna.samplers.TPESampler):
    def get_startup_trials(self) -> int:
        """Get a number of startup trials in TPESampler.

        Returns:
            int: A number of startup trials.
        """
        return self._n_startup_trials


class TpeOptimizer(AbstractOptimizer):
    """An optimizer class based on optuna.samplers.TPESampler.

    Args:
        options (dict[str, str | int | bool]): A dictionary
            containing command line options.

    Attributes:
        parameter_pool (dict[int, list[dict[str, float | int | str]]]):
            A dictionary of parameters.
        parameter_list (list[HyperParameter]): A list of HyperParameter
            objects.
        study_name (str): Study name.
        study (str): optuna.study.Study object.
        distributions (dict[str, optuna.distributions]): A dictionary of
            optuna.distributions objects defined for individual parameters.
        trial_pool (dict[int, optuna.trial.Trial]): A dictionary of
            optuna.trial.Trial objects.
        randseed (int): Random seed.
    """

    def __init__(self, options: dict[str, str | int | bool]) -> None:
        super().__init__(options)
        self.parameter_pool: dict = {}
        self.parameter_list: list = []
        self.study_name = "distributed-tpe"
        self.study: Any = None
        self.distributions: dict
        self.trial_pool: dict = {}
        self.randseed = self.config.randseed.get()

    def pre_process(self) -> None:
        """Pre-Procedure before executing optimize processes.
        """
        super().pre_process()

        self.parameter_list = self.params.get_parameter_list()
        self.create_study()
        self.distributions = create_distributions(self.params)

    def post_process(self) -> None:
        """Post-procedure after executed processes.
        """
        self.check_result()
        super().post_process()

    def check_result(self) -> None:
        """Check the result files and add it to sampler object.

        Returns:
            None
        """

        del_keys = []
        for trial_id, param in self.parameter_pool.items():
            objective = self.storage.result.get_any_trial_objective(trial_id)
            if objective is not None:
                trial = self.trial_pool[trial_id]
                self.study.tell(trial, objective)
                del_keys.append(trial_id)

        for key in del_keys:
            self.parameter_pool.pop(key)
            self.logger.info(f'trial_id {key} is deleted from parameter_pool')

        self.logger.debug(f'current pool {[k for k, v in self.parameter_pool.items()]}')

    def is_startup_trials(self) -> bool:
        """Is a current trial startup trial or not.

        Returns:
            bool: Is a current trial startup trial or not.
        """
        n_startup_trials = self.study.sampler.get_startup_trials()
        return self.num_of_generated_parameter < n_startup_trials

    def generate_parameter(
        self,
        number: int | None = 1
    ) -> list[dict[str, float | int | str]] | None:
        """Generate parameters.

        Args:
            number (int | None, optional): A number of generating parameters.
                Defaults to 1.

        Returns:
            list[dict[str, float | int | str]] | None: A list of created
            parameters.
        """
        self.check_result()
        self.logger.debug(
            f'number: {number}, pool: {len(self.parameter_pool)} losses'
        )

        # TPE has to be sequential.
        if (
            (not self.is_startup_trials()) and
            (len(self.parameter_pool) >= 1)
        ):
            return None

        if len(self.parameter_pool) >= self.config.num_node.get():
            return None

        new_params = []
        trial = self.study.ask(self.distributions)

        for param in self.params.get_parameter_list():
            new_param = {
                'parameter_name': param.name,
                'type': param.type,
                'value': trial.params[param.name]
            }
            new_params.append(new_param)

        trial_id = self.trial_id.get()
        self.parameter_pool[trial_id] = new_params
        self.trial_pool[trial_id] = trial
        self.logger.info(f'newly added name: {trial_id} to parameter_pool')

        return new_params

    def generate_initial_parameter(
        self
    ) -> list[dict[str, float | int | str]] | None:
        """Generate initial parameters.

        Returns:
            list[dict[str, float | int | str]] | None: A list of new
            parameters. None if `self.nelder_mead` is already defined.
        """
        enqueue_trial = {}
        for hp in self.params.hps.values():
            if hp.initial is not None:
                enqueue_trial[hp.name] = hp.initial

        # all hp.initial is None
        if len(enqueue_trial) == 0:
            return self.generate_parameter()

        self.study.enqueue_trial(enqueue_trial)
        trial = self.study.ask(self.distributions)

        new_params = []

        for name, value in trial.params.items():
            new_param = {
                'parameter_name': name,
                'type': self.params.hps[name].type,
                'value': value
            }
            new_params.append(new_param)

        trial_id = self.trial_id.get()
        self.parameter_pool[trial_id] = new_params
        self.trial_pool[trial_id] = trial
        self.logger.info(f'newly added name: {trial_id} to parameter_pool')
        return new_params

    def create_study(self) -> None:
        """Create the optuna.study object and store it.

        Returns:
            None
        """
        if self.study is None:
            self.study = optuna.create_study(
                sampler=TPESamplerWrapper(seed=self.randseed),
                study_name=self.study_name,
                direction=self.config.goal.get().lower()
            )


def create_distributions(
        parameters: aiaccel.parameter.HyperParameterConfiguration
) -> dict:
    """Create an optuna.distributions dictionary for the parameters.

    Args:
        parameters(aiaccel.parameter.HyperParameterConfiguration): A
            parameter configuration object.

    Raises:
        ValueError: Occurs when parameter type is other than 'float', 'int',
            'categorical', or 'ordinal'.

    Returns:
        (dict): An optuna.distributions object.
    """
    distributions: dict[str, optuna.distributions.BaseDistribution] = {}

    for p in parameters.get_parameter_list():
        if p.type.lower() == 'float':
            distributions[p.name] = optuna.distributions.FloatDistribution(
                p.lower, p.upper, log=p.log
            )

        elif p.type.lower() == 'int':
            distributions[p.name] = optuna.distributions.IntDistribution(
                p.lower, p.upper, log=p.log
            )

        elif p.type.lower() == 'categorical':
            distributions[p.name] = optuna.distributions.CategoricalDistribution(p.choices)

        elif p.type.lower() == 'ordinal':
            distributions[p.name] = optuna.distributions.CategoricalDistribution(p.sequence)

        else:
            raise TypeError('Unsupported parameter type')

    return distributions
