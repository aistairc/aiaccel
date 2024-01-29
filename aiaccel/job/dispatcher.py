from __future__ import annotations

import sys
import time

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Callable

from optuna.study import Study
from optuna.trial import Trial, TrialState

from aiaccel.job.env import Abci, Local
from aiaccel.job.parameter import Parameter, suggest_hyperparameter

script_name = sys.argv[0]

hp_args = {}

parser = ArgumentParser()
parser.add_argument(
    "-e", action="store_true", required=False, help="Execute the objective function."
)
parser.add_argument(
    "--params",
    nargs="+",
    required=False,
    help="Hyperparameter values to override the default values.",
)  # --params key1=value key2=value ...
args = parser.parse_known_args()[0]
if args.params:
    for option in args.params:
        key, value = option.split("=")
        hp_args[key] = value


__local__ = "local"
__abci__ = "abci"


class JobCreator:
    def __init__(
        self, trial_id: int,
        platform: str,
        group: str,
        preamble: str,
        timeout_seconds: int,
        work_dir: Path,
    ):
        self.trial_id = trial_id
        self.platform = platform
        self.group = group
        self.preamble = preamble
        self.work_dir = work_dir
        self.timeout_seconds = timeout_seconds
        self._start_time = None
        self._end_time = None
        self._returncode = None

        self.job_file_path = str(self.work_dir / f"job{self.trial_id}.sh")
        self.stdout_file_path = str(self.work_dir / f"o{self.trial_id}")
        self.stderr_file_path = str(self.work_dir / f"e{self.trial_id}")

        if self.platform == __local__:
            self.excution_environment = Local(
                script_name,
                self.preamble,
                self.job_file_path,
                self.stdout_file_path,
                self.stderr_file_path
            )
        elif self.platform == __abci__:
            self.excution_environment = Abci(
                script_name,
                self.preamble,
                self.group,
                self.job_file_path,
                self.work_dir,
                self.work_dir,
            )
        else:
            raise NotImplementedError(f"Platform '{self.platform}' not implemented.")

    def create(self, trial: Trial, param: Parameter) -> None:
        """Create a executable file to run the job."""
        self.excution_environment.create(trial, param)

    def run(self) -> None:
        """Run the job with the given hyperparameters."""
        self._start_time = time.time()
        self._returncode = self.excution_environment.run()
        self._end_time = time.time()

    def collect_result(self) -> str | None:
        """Collect the result of the job."""
        return self.excution_environment.collect_result()

    # ========================
    # check status
    # ========================

    def is_finished(self) -> bool:
        """Check if the job finished."""
        return self._returncode is not None

    def is_error_free(self) -> bool:
        """Check if the job finished without error."""
        return self._returncode == 0

    # ========================
    # properties
    # ========================

    @property
    def returncode(self) -> int | None:
        """Get the return code of the job."""
        return self._returncode

    # ========================
    # setter/getter
    # ========================

    def get_elapsed_time(self) -> float:
        """Get the elapsed time."""
        if self._start_time is None:
            raise RuntimeError("Job not started.")
        return time.time() - self._start_time


class JobDispatcher:
    def __init__(
        self,
        study: Study,
        func: Callable[[Trial], Any],
        parameter: Parameter,
        platform: str = "local",
        group: str = "",
        n_jobs: int = 1,
        retry_num: int = 0,
        preamble: str = "",
        timeout_seconds: int = -1,
        work_dir: str = "./",
    ):
        self.study = study
        self.func = func
        self.parameter = parameter
        self.platform = platform.lower()
        self.group = group
        self.n_jobs = n_jobs
        self.preamble = preamble
        self.retry_num = retry_num
        self.timeout_seconds = timeout_seconds
        self.work_dir = Path(work_dir).resolve()

        if not self.work_dir.exists():
            self.work_dir.mkdir(parents=True)

    ...

    # ========================
    # manage resources
    # ========================

    def running_job_count(self) -> int:
        return sum(trial.state == TrialState.RUNNING for trial in self.study.trials)

    def get_running_trial_ids(self) -> list[int]:
        return [trial._trial_id for trial in self.study.trials if trial.state == TrialState.RUNNING]

    def get_required_parallel_num(self) -> int:
        """Get the number of parallel jobs required to run the next trial for the study."""
        ...
        return 1

    def get_abvailable_worker_count(self) -> int:
        return min((self._n_jobs - self.running_job_count()), self.get_required_parallel_num())

    def wait_for_finished_job(self, trial_ids: list) -> None:
        """Wait for the specified jobs to finish."""
        ...

    def retry_failed_job(self, trial_ids: list) -> None:
        """Retry the specified jobs."""
        if self.retry_num == 0:
            return
        ...

    # ========================
    # ask/tell
    # ========================

    def ask(self) -> Trial:
        """Ask the study for a new trial."""
        return self.study.ask()

    def tell(self, trial: Trial, objective_value: Any) -> None:
        """Tell the study the objective value for a specific trial."""
        self.study.tell(trial, objective_value)

    # ========================
    # optimize
    # ========================

    def optimize(self, n_trials: int) -> None:
        """Optimize the objective function for a specified number of trials."""
        for _ in range(n_trials):
            trial = self.study.ask()
            objective_value = self.run(trial)
            self.study.tell(trial, objective_value)

    def optimize_parallel(self, n_trials: int) -> None:
        """Optimize the objective function in parallel. (To be implemented)"""
        ...

    # ========================
    # run job
    # ========================

    def _run_objective(self, params: Parameter) -> Any:
        """Run the objective function."""
        return self.func(params)

    def run(self, trial: Trial) -> float:
        """Run the job with the specified hyperparameters and collect the objective value."""
        if args.e:
            # Called by the job script file (***.sh), not invoked by the job dispatcher.
            # Retrieve hyperparameter values from command-line arguments and update the parameter object.
            # Execute the objective function and print the result.
            for name, value in hp_args.items():
                self.parameter.update_values(name, value)
            y = self._run_objective(self.parameter)
            sys.stdout.write(f"{str(y)}\n")
            sys.exit(0)
        else:
            # Create and run the job.
            job = JobCreator(trial._trial_id, self.platform, self.group, self.preamble,self.timeout_seconds,  self.work_dir)
            params = suggest_hyperparameter(trial, self.parameter)
            job.create(trial, params)  # create job file (***.sh)
            job.run()
            y = job.collect_result()
            return float(y)

    ...


"""

``` bash
python ex01.py
```

- ex01.py
``` python

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
    param = Parameter([
        {"name": "x1", "type": "float", "bounds": [0, 10], "step": 1, "log": False},
        {"name": "x2", "type": "float", "bounds": [0, 10], "step": 1, "log": False},
    ])

    study = aiaccel.create_study(direction='minimize', sampler=NelderMeadSampler(search_space=param.bounds, seed=42))
    jobs = aiaccel.JobDispatcher(study, func=objective, parameter=param, platform="local", group="", work_dir="./work")
    for _ in range(10):
        trial = jobs.ask()
        y = jobs.run(trial)
        jobs.tell(trial, y)
```
"""
