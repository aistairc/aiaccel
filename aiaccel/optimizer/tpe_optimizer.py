import copy
from typing import Optional

import numpy as np
import optuna
from optuna.trial import TrialState

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
    def __init__(self, options: dict) -> None:
        """Initial method of TpeOptimizer.

        Args:
            options (dict): A file name of a configuration.
        """
        super().__init__(options)
        self.parameter_pool = {}
        self.parameter_list = []
        self.study_name = "distributed-tpe"
        self.study = None
        self.distributions = None
        self.trial_pool = {}
        self.randseed = self.config.randseed.get()
        self.initial_count = 0

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
        results = self.storage.result.get_result_trial_id_list()
        for trial_id, params in self.parameter_pool.items():
            if int(trial_id) in results:
                trial_id_str = self.get_zero_padding_any_trial_id(trial_id)
                result_content = self.storage.get_hp_dict(trial_id_str)
                objective = result_content['result']
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
        # If initial exists, the output of n_startup_trials is reduced
        # by the number of initials before the original n_startup_trials,
        # so the number of initials needs to be added to the right side.
        return self.num_of_generated_parameter < n_startup_trials + self.initial_count

    def generate_parameter(self, number: Optional[int] = 1) -> None:
        """Generate parameters.

        Args:
            number (Optional[int]): A number of generating parameters.
        """
        self.check_result()
        self.logger.debug(f'number: {number}, pool: {len(self.parameter_pool)} losses')

        initial_parameter = self.generate_initial_parameter()

        if initial_parameter is not None:
            trial_id = self.register_ready(initial_parameter)
            self.parameter_pool[trial_id] = initial_parameter['parameters']
            self.logger.info(f'newly added name: {trial_id} to parameter_pool')

            number -= 1

        for _ in range(number):
            # TPE has to be sequential.
            if (
                (not self.is_startup_trials()) and
                (len(self.parameter_pool) >= 1)
            ):
                break

            if len(self.parameter_pool) >= self.config.num_node.get():
                break

            new_params = []
            trial = self.study.ask(self.distributions)

            for param in self.params.get_parameter_list():
                new_param = {
                    'parameter_name': param.name,
                    'type': param.type,
                    'value': trial.params[param.name]
                }
                new_params.append(new_param)

            self.num_of_generated_parameter += 1
            trial_id = self.register_ready({'parameters': new_params})
            self.parameter_pool[trial_id] = new_params
            self.trial_pool[trial_id] = trial
            self.logger.info(f'newly added name: {trial_id} to parameter_pool')

            self._serialize(self.trial_id.integer)

    def generate_initial_parameter(self):

        if self.num_of_generated_parameter == 0:
            enqueue_trial = {}
            for hp in self.params.hps.values():
                if hp.initial is not None:
                    enqueue_trial[hp.name] = hp.initial

            # all hp.initial is None
            if len(enqueue_trial) == 0:
                return None

            self.study.enqueue_trial(enqueue_trial)
            t = self.study.ask(self.distributions)
            self.initial_count += 1
            self.trial_pool[self.initial_count] = t

            new_params = []

            for name, value in t.params.items():
                new_param = {
                    'parameter_name': name,
                    'type': self.params.hps[name].type,
                    'value': value
                }
                new_params.append(new_param)

            self.num_of_generated_parameter += 1
            return {'parameters': new_params}

        return None

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

    def _serialize(self, trial_id: int) -> None:
        """Serialize this module.
        Returns:
            dict: The serialized objects.
        """
        parameter_pool = copy.deepcopy(self.parameter_pool)
        for _, params in parameter_pool.items():
            for param in params:
                if type(param['value']) is np.float64:
                    param['value'] = float(param['value'])

        # TODO: add serialize trial_pool

        self.storage.variable.d['native_random_state'].set(trial_id=trial_id, value=self.get_native_random_state())
        self.storage.variable.d['numpy_random_state'].set(trial_id=trial_id, value=self.get_numpy_random_state())
        self.storage.variable.d['num_of_generated_parameter'].set(trial_id=trial_id, value=self.num_of_generated_parameter)
        self.storage.variable.d['loop_count'].set(trial_id=trial_id, value=self.loop_count)
        self.storage.variable.d['parameter_pool'].set(trial_id=trial_id, value=parameter_pool)
        self.storage.variable.d['study'].set(trial_id=trial_id, value=self.study)

    def _deserialize(self, trial_id: int) -> None:
        """Deserialize this module.
        Args:
            dict_objects(dict): A dictionary including serialized objects.
        Returns:
            None
        """
        self.set_native_random_state(self.storage.variable.d['native_random_state'].get(trial_id))
        self.set_numpy_random_state(self.storage.variable.d['numpy_random_state'].get(trial_id))
        self.num_of_generated_parameter = self.storage.variable.d['num_of_generated_parameter'].get(trial_id)
        self.loop_count = self.storage.variable.d['loop_count'].get(trial_id)
        self.study = self.storage.variable.d['study'].get(trial_id)

        parameter_pool = copy.deepcopy(self.storage.variable.d['parameter_pool'].get(trial_id))
        for _, params in parameter_pool.items():
            for param in params:
                if type(param['value']) is float:
                    param['value'] = np.float64(param['value'])

        self.parameter_pool = parameter_pool

        # TODO: add deserialize trial_pool
        running_trials = self.study.get_trials(states=(TrialState.RUNNING,))

        for t in running_trials:
            self.trial_pool[t._trial_id + 1] = optuna.trial.Trial(self.study, t._trial_id)


def create_distributions(
        parameters: aiaccel.parameter.HyperParameterConfiguration
) -> dict:
    """Create an optuna.distributions dictionary for the parameters.

    Args:
        parameters(aiaccel.parameter.HyperParameterConfiguration): A
            parameter configuration object.

    Returns:
        (dict): An optuna.distributions object.
    """
    distributions = {}

    for p in parameters.get_parameter_list():
        if p.type == 'FLOAT':
            if p.log:
                distributions[p.name] = optuna.distributions.LogUniformDistribution(p.lower, p.upper)
            else:
                distributions[p.name] = optuna.distributions.UniformDistribution(p.lower, p.upper)

        elif p.type == 'INT':
            if p.log:
                distributions[p.name] = optuna.distributions.IntLogUniformDistribution(p.lower, p.upper)
            else:
                distributions[p.name] = optuna.distributions.IntUniformDistribution(p.lower, p.upper)

        elif p.type == 'CATEGORICAL':
            distributions[p.name] = optuna.distributions.CategoricalDistribution(p.choices)
        else:
            raise 'Unsupported parameter type'

    return distributions
