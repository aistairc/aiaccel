from typing import Any

import argparse
from collections.abc import Callable
from pathlib import Path
import pickle as pkl

from hydra.utils import instantiate
from omegaconf import OmegaConf as oc  # noqa: N813

from optuna.trial import Trial

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_filename", type=Path, help="The shell script to execute.")
    parser.add_argument("--config", nargs="?", default=None)
    parser.add_argument("--executor", nargs="?", default="local")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument(
        "--fix", nargs="*", default=[], help="Parameter names to fix with best values from previous study"
    )
    args, unk_args = parser.parse_known_args()

    config = oc.merge(oc.load(args.config), oc.from_cli(unk_args))
    config.study.setdefault(
        "storage",
        {
            "_target_": "optuna.storages.RDBStorage",
            "url": "sqlite:///optuna.db",
            "engine_kwargs": {"connect_args": {"timeout": 30}},
        },
    )
    config.study.setdefault("study_name", "aiaccel_study")

    if args.resume or args.fix:
        config.study.load_if_exists = True

        prev_study = instantiate(
            {
                "_target_": "optuna.load_study",
                "storage": config.study.storage,
                "study_name": config.study.study_name,
            }
        )

        fixed_params = {}
        for param_name in args.fix:
            if param_name in prev_study.best_params:
                fixed_params[param_name] = prev_study.best_params[param_name]
            else:
                raise ValueError(f"Parameter {param_name} not found in previous study's best_params")

        if fixed_params and "sampler" in config.study:
            config.study.sampler = {
                "_target_": "optuna.samplers.PartialFixedSampler",
                "fixed_params": fixed_params,
                "base_sampler": config.study.sampler,
            }

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
            study.tell(trial, y)

            print(f"Trial #{trial.number}: Value = {y:.8f}, Parameters = {trial.params}")

            finished_job_count += 1


if __name__ == "__main__":
    main()
