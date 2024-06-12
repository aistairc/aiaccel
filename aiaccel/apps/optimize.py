import argparse
import pickle as pkl
from pathlib import Path
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
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
  _target_: aiaccel.apps.optimize.HparamsManager
  x1: [0, 1]
  x2:
    _target_: opthuna.trial.Trial.suggest_float
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
        for param_name, obj in self.params_def.items():
            if isinstance(obj, ListConfig):
                obj = oc.to_container(obj)
            if isinstance(obj, list):
                # e.g. [0, 1]
                hparams[param_name] = trial.suggest_float(param_name, *obj)
            elif isinstance(obj, Suggest):
                # e.g. {'low': 0.0, 'high': 1.0, 'log': False}
                # configで_targer_を指定した場合を想定
                hparams[param_name] = obj.suggest_hparams(trial)
            else:
                raise NotImplementedError
        return hparams


class Suggest:
    def __init__(self, **kwargs: dict[str, float]) -> None:
        self.param_def = oc.to_object(DictConfig(kwargs))

    def suggest_hparams(self, trial: Any) -> int | float | str:
        raise NotImplementedError


class SuggestFloat(Suggest):
    def suggest_hparams(self, trial):  # type: ignore
        return trial.suggest_float(**self.param_def)


class SuggestInt(Suggest):
    def suggest_hparams(self, trial):  # type: ignore
        return trial.suggest_int(**self.param_def)


class SuggestXXX(Suggest): ...


...


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="The shell script to execute.")
    parser.add_argument("--config", nargs="?", default=None)

    args, unk_args = parser.parse_known_args()
    config = oc.merge(oc.load(args.config), oc.from_cli(unk_args))

    jobs = AbciJobExecutor(Path(args.filename), config.group, n_max_jobs=config.n_max_jobs)
    study = instantiate(oc.to_container(config.study))
    params = instantiate(oc.to_container(config.params))

    result_filename_template = "{job.cwd}/{job.job_name}_result.pkl"

    finished_job_count = 0

    while finished_job_count <= config.n_trials:
        n_max_jobs = 1 if config.n_max_jobs == 1 else min(jobs.available_slots(), config.n_trials - finished_job_count)
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
