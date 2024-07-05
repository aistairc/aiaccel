import argparse
import pickle as pkl
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

from hydra.utils import instantiate
from omegaconf import OmegaConf as oc  # noqa: N813
from optuna.trial import Trial

from aiaccel.job import AbciJobExecutor

T = TypeVar('T')

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

@dataclass
class Suggest(Generic[T]):
    name: str

    def __call__(self, trial: Any) -> T:
        raise NotImplementedError


@dataclass
class Const(Suggest[T]):
    value: T

    def __call__(self, _: Any) -> T:
        return self.value


@dataclass
class SuggestFloat(Generic[T]):
    name: str
    low: float
    high: float
    step: float | None | None = None
    log: bool = False

    def __call__(self, trial: Trial) -> float:
        return trial.suggest_float(self.name, self.low, self.high, step=self.step, log=self.log)


@dataclass
class SuggestInt(Generic[T]):
    name: str
    low: int
    high: int
    step: int = 1
    log: bool = False

    def __call__(self, trial: Trial) -> int:
        return trial.suggest_int(name=self.name, low=self.low, high=self.high, step=self.step, log=self.log)


@dataclass
class SuggestXXX: ...


...


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

    args, unk_args = parser.parse_known_args()
    config = oc.merge(oc.load(args.config), oc.from_cli(unk_args))

    jobs = AbciJobExecutor(args.job_filename, config.group, n_max_jobs=config.n_max_jobs)
    study = instantiate(config.study)
    params = instantiate(config.params)

    result_filename_template = "{job.cwd}/{job.job_name}_result.pkl"

    finished_job_count = 0

    while finished_job_count < config.n_trials:
        n_max_jobs = min(jobs.available_slots(), config.n_trials - finished_job_count)
        for _ in range(n_max_jobs):
            trial = study.ask()

            hparams = params.suggest_hparams(trial)

            jobs.job_name = str(jobs.job_filename) + f"_{trial.number}"

            job = jobs.submit(
                args=[result_filename_template] + sum([[f"--{k}", f"{v:.5f}"] for k, v in hparams.items()], []),
                tag=trial,
            )

        for job in jobs.collect_finished():
            trial = job.tag

            with open(result_filename_template.format(job=job), "rb") as f:
                y = pkl.load(f)

            study.tell(trial, y)

            finished_job_count += 1


if __name__ == "__main__":
    main()
