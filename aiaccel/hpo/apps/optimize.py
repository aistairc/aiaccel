from typing import Any

import argparse
from collections.abc import Callable
import importlib.resources
from pathlib import Path
import time

from hydra.utils import instantiate
from omegaconf import OmegaConf as oc  # noqa: N813

from optuna.trial import Trial

from aiaccel.hpo.optuna.suggest_wrapper import Const, Suggest, SuggestFloat, T
from aiaccel.utils import print_config


class HparamsManager:
    """
    Manages hyperparameters for optimization.
    This class allows defining hyperparameters with various types and provides
    a method to suggest hyperparameters for a given trial.
    Attributes:
        params (dict): A dictionary where keys are hyperparameter names and values
                       are callables that take a Trial object and return a hyperparameter value.
    Methods:
        __init__(**params_def: dict[str, int | float | str | list[int | float] | Suggest[T]]) -> None:
            Initializes the HparamsManager with the given hyperparameter definitions.
        suggest_hparams(trial: Trial) -> dict[str, float | int | str | list[float | int | str]]:
            Suggests hyperparameters for the given trial.
    """

    def __init__(self, **params_def: dict[str, int | float | str | list[int | float] | Suggest[T]]) -> None:
        self.params: dict[str, Callable[[Trial], Any]] = {}
        for name, param in params_def.items():
            if callable(param):
                self.params[name] = param
            else:
                if isinstance(param, list):
                    low, high = param
                    self.params[name] = SuggestFloat(name=name, low=low, high=high)
                else:
                    self.params[name] = Const(name=name, value=param)

    def suggest_hparams(self, trial: Trial) -> dict[str, float | int | str | list[float | int | str]]:
        """
        Suggests hyperparameters for a given trial.
        This method generates a dictionary of hyperparameters by applying the
        parameter functions stored in `self.params` to the provided trial.
        Args:
            trial (Trial): An Optuna trial object used to suggest hyperparameters.
        Returns:
            dict[str, float | int | str | list[float | int | str]]: A dictionary
            where keys are parameter names and values are the suggested
            hyperparameters, which can be of type float, int, str, or a list of
            these types.
        """

        return {name: param_fn(trial) for name, param_fn in self.params.items()}


def main() -> None:
    """
    Main function to execute the hyperparameter optimization process using a Dask cluster.
    This function parses command-line arguments, loads the configuration,
    sets up the Dask client, and runs the optimization trials in a distributed manner.

    Command-line arguments:
        - --config (str, optional): Path to the configuration file.
        - --resume (bool, optional): Flag to resume from the previous study.
        - --resumable (bool, optional): Flag to make the study resumable by setting appropriate storage.

    Usage:
        - Start a new study:
            python -m aiaccel.hpo.apps.optimize --config config.yaml
        - Resume from the previous study:
            python -m aiaccel.hpo.apps.optimize --config config.yaml --resume
        - Make the study resumable:
            python -m aiaccel.hpo.apps.optimize --config config.yaml --resumable

    Config file (yaml) example:
        ~~~ yaml
        study:
          _target_: optuna.create_study
          direction: minimize
          storage:
            _target_: optuna.storages.InMemoryStorage
          study_name: aiaccel_study
          load_if_exists: false

        cluster:
          _target_: distributed.Client
          n_workers: 4
          threads_per_worker: 1

        sampler:
          _target_: optuna.samplers.TPESampler
          seed: 0

        params:
          _convert_: partial
          _target_: aiaccel.apps.optimize.HparamsManager
          x1: [0, 1]
          x2:
            _target_: aiaccel.apps.optimize.SuggestFloat
            name: x2
            low: 0.0
            high: 1.0
            log: false

        n_trials: 30
        n_max_jobs: 4
        ~~~
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--resumable", action="store_true", default=False)

    args, unk_args = parser.parse_known_args()

    default_config = oc.load(importlib.resources.open_text("aiaccel.hpo.apps.config", "default.yaml"))
    config = oc.merge(default_config, oc.load(args.config) if args.config is not None else {})
    config = oc.merge(config, oc.from_cli(unk_args))

    if (args.resumable or args.resume) and ("storage" not in config.study or args.config is None):
        config = oc.merge(config, oc.load(importlib.resources.open_text("aiaccel.hpo.apps.config", "resumable.yaml")))

    if args.resume:
        config.study.load_if_exists = True

    print_config(config)

    client = instantiate(config.cluster)
    client.cluster.scale(config.n_max_jobs)

    work_dir = Path.cwd()
    work_dir.mkdir(parents=True, exist_ok=True)

    study = instantiate(config.study)
    params = instantiate(config.params)
    objective_func = instantiate(config.objective, _partial_=True)

    future_to_trial: dict[Any, dict[str, Any]] = {}
    submitted_job_count = 0
    finished_job_count = 0

    try:
        while finished_job_count < config.n_trials:
            active_jobs = len(list(future_to_trial.keys()))
            available_slots = max(0, config.n_max_jobs - active_jobs)
            n_to_submit = min(available_slots, config.n_trials - submitted_job_count)

            for _ in range(n_to_submit):
                trial = study.ask()
                hparams = params.suggest_hparams(trial)
                future = client.submit(objective_func, **hparams)
                future_to_trial[future] = {"trial": trial}
                submitted_job_count += 1

            for future in list(future_to_trial.keys()):
                if future.done():
                    trial_info = future_to_trial.pop(future)
                    trial = trial_info["trial"]
                    y = future.result()
                    study._log_completed_trial(study.tell(trial, y))
                    finished_job_count += 1
                    continue
                else:
                    time.sleep(0.5)

            if n_to_submit == 0 and active_jobs == len([f for f in future_to_trial if not f.done()]):
                time.sleep(0.5)

    finally:
        client.close()


if __name__ == "__main__":
    main()
