from typing import Any

import argparse
from collections.abc import Callable
import importlib.resources
from pathlib import Path
import pickle as pkl

from hydra.utils import instantiate
from omegaconf import OmegaConf as oc  # noqa: N813

from optuna.trial import Trial

from aiaccel.hpo.job_executors import AbciJobExecutor, BaseJobExecutor, LocalJobExecutor
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
    Main function to execute the hyperparameter optimization process.
    This function parses command-line arguments, loads the configuration,
    sets up the job executor, and runs the optimization trials.

    Command-line arguments:
        - job_filename (Path): The shell script to execute.
        - --config (str, optional): Path to the configuration file.
        - --executor (str, optional): Type of job executor to use ("local" or "abci").
        - --resume (bool, optional): Flag to resume from the previous study.

    The function performs the following steps:
        1. Parses command-line arguments.
        2. Loads and merges the configuration from the file and command-line arguments.
        3. Sets default storage and study name if not provided in the configuration.
        4. Initializes the job executor based on the specified executor type.
        5. Instantiates the study and parameter suggestion objects.
        6. Submits jobs for hyperparameter optimization trials.
        7. Collects and processes finished jobs, updating the study with results.

    Usage:
        python -m aiaccel.hpo.apps.optimize objective.sh --config config.yaml

    Config file (yaml) example:
        ~~~ yaml
        study:
        _target_: optuna.create_study
        direction: minimize

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

        group: gaa50000
        ~~~
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("job_filename", type=Path, help="The shell script to execute.")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--executor", nargs="?", default="local")
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

    jobs: BaseJobExecutor

    if args.executor.lower() == "local":
        jobs = LocalJobExecutor(args.job_filename, n_max_jobs=config.n_max_jobs)
    elif args.executor.lower() == "abci":
        jobs = AbciJobExecutor(args.job_filename, config.group, n_max_jobs=config.n_max_jobs)
    else:
        raise ValueError(f"Unknown executor: {args.executor}")

    study = instantiate(config.study)
    params = instantiate(config.params)

    result_filename_template = "{job.cwd}/{job.job_name}_result.pkl"

    n_running_jobs = 0
    finished_job_count = 0

    while finished_job_count < config.n_trials:
        # n_running_jobs = len(jobs.get_running_jobs())
        n_max_jobs = min(jobs.available_slots(), config.n_trials - finished_job_count - n_running_jobs)
        for _ in range(n_max_jobs):
            trial = study.ask()

            hparams = params.suggest_hparams(trial)

            jobs.job_name = str(jobs.job_filename) + f"_{trial.number}"

            job = jobs.submit(
                args=[result_filename_template] + sum([[f"--{k}", f"{v:.5f}"] for k, v in hparams.items()], []),
                tag=trial,
            )
            n_running_jobs += 1

        for job in jobs.collect_finished():
            trial = job.tag

            with open(result_filename_template.format(job=job), "rb") as f:
                y = pkl.load(f)

            study.tell(trial, y)

            n_running_jobs -= 1
            finished_job_count += 1


if __name__ == "__main__":
    main()
