import optuna

import aiaccel.parameter
from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer
from aiaccel.optimizer.tpe.sampler import TPESamplerWrapper
from optuna.trial import TrialState
from typing import Optional
import aiaccel.util.filesystem as fs
import copy
import joblib
import numpy as np
import pathlib


class TpeSearchOptimizer(AbstractOptimizer):
    def __init__(self, options: dict) -> None:
        """Initial method of Optimizer.

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
        result_files = fs.get_file_result(self.ws, self.dict_lock)
        hashnames = [fs.get_basename(f) for f in result_files]

        for hashname, params in self.parameter_pool.items():
            try:
                index = hashnames.index(hashname)
            except ValueError:
                continue
            result_file = result_files[index]
            result_content = fs.load_yaml(result_file, self.dict_lock)
            result = result_content['result']
            trial = self.trial_pool[hashname]
            self.study.tell(trial, result)
            del_keys.append(hashname)

        for key in del_keys:
            self.parameter_pool.pop(key)
            self.logger.info(
                'hashname {} is deleted from parameter_pool'
                .format(key)
            )

        self.logger.debug(
            'current pool {}'
            .format(
                [k for k, v in self.parameter_pool.items()]
            )
        )

    def is_startup_trials(self) -> bool:
        """Is a current trial startup trial or not.

        Returns:
            bool: Is a current trial startup trial or not.
        """
        n_startup_trials = self.study.sampler.get_startup_trials()
        return self.generated_parameter < n_startup_trials

    def generate_parameter(self, number: Optional[int] = 1) -> None:
        """Generate parameters.

        Args:
            number (Optional[int]): A number of generating parameters.
        """
        self.check_result()
        self.logger.debug(
            'number: {}, pool: {} losses'
            .format(number, len(self.parameter_pool))
        )

        initial_parameter = self.generate_initial_parameter()

        if initial_parameter is not None:
            hashname = self.create_parameter_file(initial_parameter)
            self.parameter_pool[hashname] = initial_parameter['parameters']
            enqueue_trial = {}

            for param in self.parameter_pool[hashname]:
                enqueue_trial[param['parameter_name']] = param['value']

            self.study.enqueue_trial(enqueue_trial)
            self.logger.info(
                'newly added name: {} to parameter_pool'.format(hashname)
            )
            t = self.study.ask(self.distributions)
            self.trial_pool[hashname] = t
            self.generated_parameter += 1
            number -= 1

        for n in range(number):
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

            hashname = self.create_parameter_file({'parameters': new_params})
            self.parameter_pool[hashname] = new_params
            self.trial_pool[hashname] = trial
            self.logger.info(
                'newly added name: {} to parameter_pool'
                .format(hashname)
            )
            self.generated_parameter += 1

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

    @property
    def study_pickle_path(self) -> pathlib.Path:
        """Returns the path object to store the pickled optuna.study object.

        Returns:
            pathlib.Path: the path object to store the pickled optuna.study
            object.
        """
        return self.ws / 'state' / str(self.curr_trial_number) / 'study.pkl'

    def _serialize(self) -> dict:
        """Serialize this module.

        Returns:
            dict: The serialized objects.
        """
        parameter_pool = copy.deepcopy(self.parameter_pool)
        for hashname, params in parameter_pool.items():
            for param in params:
                if type(param['value']) is np.float64:
                    param['value'] = float(param['value'])

        # TODO: add serialize trial_pool

        self.serialize_datas = {
            'generated_parameter': self.generated_parameter,
            'loop_count': self.loop_count,
            'parameter_pool': parameter_pool
        }
        super()._serialize()
        # TODO: add the path to pickled study.
        base_dir_path = self.ws / 'state' / str(self.curr_trial_number)
        if base_dir_path.exists():
            joblib.dump(self.study, str(self.study_pickle_path))
        return {}

    def _deserialize(self, dict_objects: dict) -> None:
        """Deserialize this module.

        Args:
            dict_objects(dict): A dictionary including serialized objects.

        Returns:
            None
        """
        super()._deserialize(dict_objects)
        parameter_pool = copy.deepcopy(dict_objects['parameter_pool'])
        for hashname, params in parameter_pool.items():
            for param in params:
                if type(param['value']) is float:
                    param['value'] = np.float64(param['value'])

        self.parameter_pool = parameter_pool
        self.study = joblib.load(str(self.study_pickle_path))
        # TODO: add deserialize trial_pool
        running_files = fs.get_file_hp_running(self.ws, self.dict_lock)
        running_trials = self.study.get_trials(states=(TrialState.RUNNING,))

        for _ in range(len(running_trials)):
            t = self.study.ask(self.distributions)

            for running_file in running_files:
                rf_content = fs.load_yaml(running_file)
                rf_param_dict = {
                    i['parameter_name']: i['value']
                    for i in rf_content['parameters']
                }
                match = True

                for k, v in t.params.items():
                    if rf_param_dict[k] != v:
                        match = False
                        break
                if match:
                    self.trial_pool[rf_content['hashname']] = t
                    break
            else:
                # debug
                print('Running trial does not match any running files.')
                print('\ttrial params: ', t.params)
                raise ()


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
                distributions[p.name] = \
                    optuna.distributions.LogUniformDistribution(
                    p.lower,
                    p.upper
                )
            else:
                distributions[p.name] = \
                    optuna.distributions.UniformDistribution(
                    p.lower,
                    p.upper
                )
        elif p.type == 'INT':
            if p.log:
                distributions[p.name] = \
                    optuna.distributions.IntLogUniformDistribution(
                    p.lower,
                    p.upper
                )
            else:
                distributions[p.name] = \
                    optuna.distributions.IntUniformDistribution(
                    p.lower,
                    p.upper
                )
        elif p.type == 'CATEGORICAL':
            distributions[p.name] = \
                optuna.distributions.CategoricalDistribution(
                    p.choices
            )
        else:
            raise 'Unsupported parameter type'

    return distributions
