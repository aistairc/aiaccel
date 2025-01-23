from typing import Any

import argparse
from collections.abc import Callable
import importlib.resources
from pathlib import Path

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
    Main function to execute the hyperparameter optimization process.
    This function parses command-line arguments, loads the configuration,
    sets up the job executor, and runs the optimization trials.

    Command-line arguments:
        - job_filename (Path): The shell script to execute.
        - --config (str, optional): Path to the configuration file.
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

        executor:
            _target_: aiaccel.hpo.job_executors.LocalJobExecutor
            n_max_jobs: 4

        result:
            _target_: aiaccel.hpo.results.JsonResult
            filename_template: "{job.cwd}/{job.job_name}_result.json"

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
        ~~~
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("job_filename", type=Path, help="The shell script to execute.")
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

    config.executor.job_filename = args.job_filename

    jobs = instantiate(config.executor)
    study = instantiate(config.study)
    params = instantiate(config.params)
    result = instantiate(config.result)

    finished_job_count = 0

    while finished_job_count < config.n_trials:
        n_running_jobs = len(jobs.get_running_jobs())
        n_max_jobs = min(jobs.available_slots(), config.n_trials - finished_job_count - n_running_jobs)
        for _ in range(n_max_jobs):
            trial = study.ask()

            hparams = params.suggest_hparams(trial)

            jobs.job_name = str(jobs.job_filename) + f"_{trial.number}"

            job = jobs.submit(
                args=[result.filename_template] + sum([[f"--{k}", f"{v}"] for k, v in hparams.items()], []),
                tag=trial,
            )

        for job in jobs.collect_finished():
            trial = job.tag

            y = result.load(job)

            study.tell(trial, y)
            print(f"Trial {trial.number} finished with value {y}")
            finished_job_count += 1


if __name__ == "__main__":
    main()
