import argparse
import pickle as pkl
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hydra.utils import instantiate
from omegaconf import OmegaConf as oc  # noqa: N813

from aiaccel.job import AbciJobExecutor

"""
Usage:
    python -m aiaccel.apps.optimize objective.sh params.x1="[0, 10]" params.x2="[0, 10]" --config config.yaml
"""

"""
config.yaml:

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
    low: 0.0
    high: 1.0
    log: false

n_trials: 30
n_max_jobs: 4

group: gaa50000

"""


class HparamsManager:
    def __init__(self, **params_def: dict[str, float]) -> None:
        self.params_def = params_def

    def suggest_hparams(self, trial: Any) -> dict[str, int | float | str] | Exception:
        hparams = {}

        for name, param in self.params_def.items():
            if isinstance(param, list):
                low, high = param
                hparams[name] = trial.suggest_float(name, low, high)
            elif isinstance(param, Suggest):
                hparams[name] = param(trial)
            else:
                raise NotImplementedError(f"Unsupported parameter type: {type(param)}")

        return hparams


@dataclass
class Suggest:
    def __call__(self, trial: Any) -> int | float | str:
        raise NotImplementedError


@dataclass
class SuggestFloat(Suggest):
    name: str
    low: float
    high: float
    step: float | None | None = None
    log: bool | None = False

    def __call__(self, trial: Any) -> float:
        return trial.suggest_float(name=self.name, low=self.low, high=self.high, step=self.step, log=self.log)  # type: ignore


@dataclass
class SuggestInt(Suggest):
    name: str
    low: int
    high: int
    step: int | None = 1
    log: bool | None = False

    def __call__(self, trial: Any) -> int:
        return trial.suggest_int(name=self.name, low=self.low, high=self.high, step=self.step, log=self.log)  # type: ignore


@dataclass
class SuggestXXX(Suggest): ...


...


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("job_filename", type=str, help="The shell script to execute.")
    parser.add_argument("--config", nargs="?", default=None)

    args, unk_args = parser.parse_known_args()
    config = oc.merge(oc.load(args.config), oc.from_cli(unk_args))

    jobs = AbciJobExecutor(Path(args.job_filename), config.group, n_max_jobs=config.n_max_jobs)
    study = instantiate(config.study)
    params = instantiate(config.params)

    result_filename_template = "{job.cwd}/{job.job_name}_result.pkl"

    finished_job_count = 0

    while finished_job_count <= config.n_trials:
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
