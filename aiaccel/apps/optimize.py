from typing import Any

import argparse
from collections.abc import Callable
from pathlib import Path
import pickle as pkl

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as oc  # noqa: N813

import optuna
from optuna.trial import Trial, TrialState

from aiaccel.hpo.optuna.suggest_wrapper import Const, Suggest, SuggestFloat, T
from aiaccel.job import AbciJobExecutor, BaseJobExecutor, LocalJobExecutor

"""
Usage (if parameters are not defined in a config file):
    python -m aiaccel.apps.optimize objective.sh params.x1="[0, 10]" params.x2="[0, 10]" --config config.yaml

Usage (if parameters are defined in a config file):
    python -m aiaccel.apps.optimize objective.sh --config config.yaml
"""

"""
config file (yaml) example:

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

"""


class HparamsManager:
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
        return {name: param_fn(trial) for name, param_fn in self.params.items()}


# NOTE: This function is extracted to reduce cyclomatic complexity of the main function
# while maintaining logical cohesion of the study configuration setup process.
def setup_study_config(config: DictConfig | ListConfig, args: argparse.Namespace) -> None:
    """Set up study configuration based on command line arguments.

    Args:
        config: Configuration object
        args: Command line arguments
    """
    default_storage = {
        "_target_": "optuna.storages.RDBStorage",
        "url": "sqlite:///study.db",
    }

    if ("study" not in config and (args.resumable or args.resume or args.fix)) or (
        args.resumable and "storage" not in config.study
    ):
        if "study" not in config:
            config.study = {"storage": default_storage}
        else:
            config.study.storage = default_storage

    if "study_name" not in config.study:
        config.study.study_name = "aiaccel_study"

    if args.resume or args.fix:
        config.study.load_if_exists = True
        storage = instantiate(config.study.storage)
        prev_study = optuna.study.load_study(storage=storage, study_name=config.study.study_name)

        if args.fix:
            fixed_params = {}
            for param_name in args.fix:
                if param_name in prev_study.best_params:
                    fixed_params[param_name] = prev_study.best_params[param_name]
                else:
                    raise ValueError(f"Parameter {param_name} not found in previous study's best_params")

            if fixed_params and "sampler" in config.study:
                base_sampler = config.study.sampler
                config.study.sampler = {
                    "_target_": "optuna.samplers.PartialFixedSampler",
                    "fixed_params": fixed_params,
                    "base_sampler": base_sampler,
                }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_filename", type=Path, help="The shell script to execute.")
    parser.add_argument("--config", nargs="?", default=None)
    parser.add_argument("--executor", nargs="?", default="local")
    parser.add_argument("--resumable", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument(
        "--fix", nargs="*", default=[], help="Parameter names to fix with best values from previous study"
    )
    args, unk_args = parser.parse_known_args()

    config = oc.merge(oc.load(args.config), oc.from_cli(unk_args))
    setup_study_config(config, args)

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

    finished_job_count = 0

    while finished_job_count < config.n_trials:
        n_running_jobs = len(jobs.get_running_jobs())
        n_max_jobs = min(jobs.available_slots(), config.n_trials - finished_job_count - n_running_jobs)

        for _ in range(n_max_jobs):
            trial = study.ask()

            hparams = params.suggest_hparams(trial)

            jobs.job_name = str(jobs.job_filename) + f"_{trial.number}"

            job = jobs.submit(
                args=[result_filename_template] + sum([[f"--{k}", f"{v}"] for k, v in hparams.items()], []),
                tag=trial,
            )

        for job in jobs.collect_finished():
            trial = job.tag

            with open(result_filename_template.format(job=job), "rb") as f:
                y = pkl.load(f)

            frozen_trial = optuna.study._tell._tell_with_warning(
                study=study,
                trial=trial,
                value_or_values=y,
                state=TrialState.COMPLETE,
                suppress_warning=True,
            )
            study._log_completed_trial(frozen_trial)

            finished_job_count += 1


if __name__ == "__main__":
    main()
