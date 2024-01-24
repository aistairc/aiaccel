from __future__ import annotations

import sys

from argparse import ArgumentParser
from pathlib import Path

from typing import Any, Callable
from optuna.trial import Trial, TrialState
from optuna.study import Study

from aiaccel.job.env import Local, Abci
from aiaccel.job.parameter import Parameter

script_name = sys.argv[0]

parser = ArgumentParser()
parser.add_argument("--e", action="store_true", help="Execute the objective function.")
args = parser.parse_args()


class JobCreator(object):
    def __init__(self, trial_id: int, platform: str, group: str, work_dir: Path):
        self.trial_id = trial_id
        self.job_name = f"job_{self.trial_id}"
        self.job_file = str(work_dir / f"{self.job_name}.sh")
        self.stdout_file = str(work_dir / f"o{self.trial_id}")
        self.stderr_file = str(work_dir / f"e{self.trial_id}")
        self.group = group

        if platform == "local":
            self.excution_environment = Local(script_name, self.job_file, self.stdout_file, self.stderr_file)
        elif platform == "abci":
            self.excution_environment = Abci(script_name,  self.group, self.job_file, self.stdout_file, self.stderr_file)
        else:
            raise NotImplementedError

    def create(self, trial: Trial):
        """Create a hyperparameter object.
        """
        self.excution_environment.create(trial)

    def suggest(self, trial: Trial, parameter: Parameter) -> Parameter:
        """Suggest a hyperparameter value.
        """
        return _suggest_hyperparameter(trial, parameter)

    def run(self) -> None | Any:
        """Run the job with the given hyperparameters.
        """
        self.excution_environment.run()

    def collect_result(self) -> Any | None:
        """Collect the result of the job.

        return:
            The result of the job (objective value).
        """
        return self.excution_environment.collect_result()


class JobDispatcher(object):
    def __init__(
        self,
        study: Study,
        func: Callable[[Trial], Any],
        parameter: Parameter,
        platform : str = "local",
        group: str = "",
        n_jobs: int = 1,
        timeout_seconds: int = -1,
        work_dir: str = "./"
    ):
        self.study = study
        self.func = func
        self.parameter = parameter
        self.platform = platform.lower()
        self.group = group
        self.n_jobs = n_jobs
        self.timeout_seconds = timeout_seconds
        self.work_dir = Path(work_dir).resolve()
        if not self.work_dir.exists():
            self.work_dir.mkdir(parents=True)

    def running_job_count(self) -> int:
        return sum(trial.state == TrialState.RUNNING for trial in self.study.trials)

    def get_abvailable_worker_count(self) -> int:
        return self._n_jobs - self.running_job_count()

    def run(self, trial: Trial) -> Any:
        """Run the job with the given hyperparameters.
        """
        if args.e:
            job = JobCreator(trial._trial_id, self.platform, self.group, self.work_dir)
            params = job.suggest(trial, self.parameter)  # suggest hyperparameters
            objective_value =  self._run_objective(params)
            sys.stdout.write(f"{str(objective_value)}\n")
            sys.exit(0)
        job = JobCreator(trial._trial_id, self.platform, self.group, self.work_dir)
        job.suggest(trial, self.parameter)  # suggest hyperparameters
        job.create(trial)   # create job file (***.sh)
        job.run()
        y = job.collect_result()
        return float(y)

    def _run_objective(self, params: Parameter) -> Any:
        """Run the objective function.
        """
        return self.func(params)

    def ask(self) -> Trial:
        """Ask the next hyperparameters.
        """
        return self.study.ask()

    def tell(self, trial: Trial, objective_value: Any) -> None:
        """Tell the result of the job.
        """
        self.study.tell(trial, objective_value)

    def optimize(self, n_trials: int) -> None:
        """Optimize the objective function.
        """
        for _ in range(n_trials):
            trial = self.study.ask()
            objective_value = self.run(trial)
            self.study.tell(trial, objective_value)


def _suggest_hyperparameter(trial: Trial, parameter: Parameter) -> Parameter:
    """Suggest hyperparameters.
    """
    for name in parameter.names:
        if parameter.type[name] == "int":
            parameter.values[name] = trial.suggest_int(name, parameter.bounds[name][0], parameter.bounds[name][1])
        elif parameter.type[name] == "float":
            parameter.values[name] = trial.suggest_float(name, parameter.bounds[name][0], parameter.bounds[name][1])
        elif parameter.type[name] == "categorical":
            parameter.values[name] = trial.suggest_categorical(name, parameter.bounds[name])
        else:
            raise NotImplementedError
    return parameter



"""
import aiaccel

from aiaccel import NelderMeadSampler, Parameter

import numpy as np


def objective(hp: Parameter) -> float:
    x1 = hp.values["x1"]
    x2 = hp.values["x2"]
    y = -20*np.exp(-0.2*np.sqrt(0.5*(x1**2+x2**2)))-np.exp(0.5 *
        (np.cos(2*np.pi*x1)+np.cos(2*np.pi*x2)))+np.e+20
    return float(y)


if __name__ == "__main__":
    param = Parameter(
        [
            {"name": "x1", "type": "float", "bounds": [0, 10], "initial": 3, "step": 1, "log": False},
            {"name": "x2", "type": "float", "bounds": [0, 10], "initial": 6, "step": 1, "log": False},
        ]
    )

    study = aiaccel.create_study(
        direction='minimize',
        sampler=NelderMeadSampler(search_space=param.bounds, seed=42)
    )

    jobs = aiaccel.JobDispatcher(study, func=objective, parameter=param, platform="local", group="", work_dir="./work")

    for _ in range(10):
        trial = jobs.ask()
        y = jobs.run(trial)
        jobs.tell(trial, y)
"""
