from __future__ import annotations

from typing import Any

import numpy
import optuna
import sqlalchemy
import sqlalchemy.orm as sqlalchemy_orm
from omegaconf.dictconfig import DictConfig
from optuna.samplers._lazy_random_state import LazyRandomState
from optuna.samplers._random import RandomSampler
from optuna.storages._rdb import models

from aiaccel.optimizer import AbstractOptimizer
from aiaccel.parameter import (
    CategoricalParameter,
    FloatParameter,
    HyperParameterConfiguration,
    IntParameter,
    OrdinalParameter,
)


class LazyRandomStateWrapper(LazyRandomState):
    def __init__(self, rng: numpy.random.RandomState | None = None) -> None:
        super().__init__()
        self._rng = rng


class RandomSamplerWrapper(RandomSampler):
    def __init__(self, rng: numpy.random.RandomState | None = None) -> None:
        super().__init__()
        self._rng = LazyRandomStateWrapper(rng)


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

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.parameter_pool: dict[str, Any] = {}
        self.study_name = "distributed-tpe"
        self.study: Any = None
        self.distributions: Any = None
        self.trial_pool: dict[str, Any] = {}
        self.randseed = self.config.optimize.rand_seed
        self.distributions = create_distributions(self.params)
        self.create_study()

    def check_result(self) -> None:
        """Check the result files and add it to sampler object.

        Args:
            None

        Returns:
            None
        """

        for trial_id in list(self.parameter_pool.keys()):
            objective = self.get_any_trial_objective(int(trial_id))

            if objective is not None:
                self.study.tell(int(trial_id), objective, skip_if_finished=True)
                del self.parameter_pool[trial_id]

    def is_startup_trials(self) -> bool:
        """Is a current trial startup trial or not.

        Args:
            None

        Returns:
            bool: Is a current trial startup trial or not.
        """
        n_startup_trials = self.study.sampler.get_startup_trials()
        return self.num_of_generated_parameter < n_startup_trials

    def generate_parameter(self, number: int | None = 1) -> list[dict[str, float | int | str]] | None:
        """Generate parameters.

        Args:
            number (int | None, optional): A number of generating parameters.
                Defaults to 1.

        Returns:
            list[dict[str, float | int | str]] | None: A list of created
            parameters.
        """

        self.check_result()
        self.logger.debug(f"generate_parameter requests {number} params, pool length: {len(self.parameter_pool)}")

        # TPE has to be sequential.
        if (not self.is_startup_trials()) and (len(self.parameter_pool) >= 1):
            return None

        if len(self.parameter_pool) >= self.config.resource.num_workers:
            return None

        new_params: list[dict[str, Any]] = []
        trial = self.study.ask(self.distributions)
        new_params = []

        for param in self.params.get_parameter_list():
            new_param = {
                "parameter_name": param.name,
                "type": param.type,
                "value": trial.params[param.name],
            }
            new_params.append(new_param)

        trial_id = self.trial_id.get()
        self.parameter_pool[trial_id] = new_params
        self.trial_pool[trial_id] = trial
        self.logger.info(f"new parameter {trial_id} is added to parameter_pool {new_params}")

        return new_params

    def generate_initial_parameter(self) -> list[dict[str, float | int | str]] | None:
        """Generate initial parameters.

        Args:
            None

        Returns:
            list[dict[str, float | int | str]] | None: A list of new
            parameters. None if `self.nelder_mead` is already defined.
        """
        enqueue_trial = {}
        for hp in self.params.get_parameter_list():
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
                "parameter_name": name,
                "type": self.params.get_hyperparameter(name).type,
                "value": value,
            }
            new_params.append(new_param)

        trial_id = self.trial_id.get()
        self.parameter_pool[trial_id] = new_params
        self.trial_pool[trial_id] = trial
        self.logger.info(f"new initial parameter {trial_id} is added to parameter_pool {new_params}")
        return new_params

    def create_study(self) -> None:
        """Create the optuna.study object and store it.

        Args:
            None

        Returns:
            None
        """

        sampler = TPESamplerWrapper()
        sampler._rng = LazyRandomStateWrapper(rng=self._rng)
        sampler._random_sampler = RandomSamplerWrapper(rng=self._rng)
        storage_path = str(f"sqlite:///{self.workspace.path}/optuna-{self.study_name}.db")
        storage = optuna.storages.RDBStorage(url=storage_path)
        load_if_exists = self.config.resume is not None
        self.study = optuna.create_study(
            sampler=sampler,
            storage=storage,
            study_name=self.study_name,
            load_if_exists=load_if_exists,
            direction=self.goals[0].lower(),
        )

    def resume_trial(self) -> None:
        """Resume the trial.

        Args:
            None

        Returns:
            None
        """
        optuna_trials = self.study.get_trials()
        storage_path = f"sqlite:///{self.workspace.path}/optuna-{self.study_name}.db"
        engine = sqlalchemy.create_engine(storage_path, echo=False)
        session = sqlalchemy_orm.sessionmaker(bind=engine)()
        for optuna_trial in optuna_trials:
            if optuna_trial.number >= self.config.resume:
                resumed_trial = session.query(models.TrialModel).filter_by(number=optuna_trial.number).first()
                session.delete(resumed_trial)
        session.commit()
        for trial_id in list(self.parameter_pool.keys()):
            objective = self.get_any_trial_objective(int(trial_id))
            if objective is not None:
                del self.parameter_pool[trial_id]
                self.logger.info(f"resume_trial trial_id {trial_id} is deleted from parameter_pool")

    def resume(self) -> None:
        """Resume the optimizer.

        Args:
            None

        Returns:
            None
        """
        super().resume()
        if self.distributions is None:
            self.distributions = create_distributions(self.params)
        if self.config.resume is not None and self.config.resume > 0:
            self.resume_trial()
        self.create_study()


def create_distributions(
    parameters: HyperParameterConfiguration,
) -> dict[str, Any]:
    """Create an optuna.distributions dictionary for the parameters.

    Args:
        parameters(HyperParameterConfiguration): A
            parameter configuration object.

    Raises:
        ValueError: Occurs when parameter type is other than 'uniform_float',uniform_int',
            'categorical', or 'ordinal'.

    Returns:
        (dict): An optuna.distributions object.
    """
    distributions: dict[str, Any] = {}

    for p in parameters.get_parameter_list():
        if isinstance(p, FloatParameter):
            distributions[p.name] = optuna.distributions.FloatDistribution(p.lower, p.upper, log=p.log)
        elif isinstance(p, IntParameter):
            distributions[p.name] = optuna.distributions.IntDistribution(p.lower, p.upper, log=p.log)
        elif isinstance(p, CategoricalParameter):
            distributions[p.name] = optuna.distributions.CategoricalDistribution(p.choices)
        elif isinstance(p, OrdinalParameter):
            distributions[p.name] = optuna.distributions.CategoricalDistribution(p.sequence)
        else:
            raise TypeError("Unsupported parameter type")
    return distributions
