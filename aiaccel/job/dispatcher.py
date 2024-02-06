from __future__ import annotations

import sys

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Callable
from aiaccel.job.job_creator import JobCreator
from concurrent.futures import ThreadPoolExecutor, as_completed


__default_work_dir__ = "./work"
__default_timeout_seconds__ = -1  # no timeout
__default_retry_num__ = 0  # no retry
__default_n_jobs__ = 1  # no parallel execution

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


class JobDispatcher:
    def __init__(
        self,
        platform: str = "local",
        group: str = "",
        preamble: str = "",
        n_jobs: int = __default_n_jobs__,
        retry_num: int = __default_retry_num__,
        timeout_seconds: int = __default_timeout_seconds__,
        work_dir: str = __default_work_dir__,
    ):
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
        ...

    def get_running_trial_ids(self) -> list[int]:
        ...

    def get_required_parallel_num(self) -> int:
        """Get the number of parallel jobs required to run the next trial for the study."""
        ...
        return 1

    def get_abvailable_worker_count(self) -> int:
        return min(
            (self._n_jobs - self.running_job_count()), self.get_required_parallel_num()
        )

    def wait_for_finished_job(self, trial_ids: list) -> None:
        """Wait for the specified jobs to finish."""
        ...

    def retry_failed_job(self, trial_ids: list) -> None:
        """Retry the specified jobs."""
        if self.retry_num == 0:
            return
        ...

    # ========================
    # job
    # ========================

    def submit(self, objective: Callable, hparams: dict, trial_id: int) -> float:
        """Run the job with the specified hyperparameters and collect the objective value."""
        if args.e:
            # Called by the job script file (***.sh), not invoked by the job dispatcher.
            # Retrieve hyperparameter values from command-line arguments and update the parameter object.
            # Execute the objective function and print the result.
            for k, v in hp_args.items():
                hparams[k] = float(v)
            y = _run_objective(objective, hparams)
            sys.stdout.write(f"{str(y)}\n")
            sys.exit(0)
        else:
            # Create and run the job.
            job = JobCreator(
                trial_id,
                self.platform,
                self.group,
                self.preamble,
                self.timeout_seconds,
                self.work_dir,
            )
            job.create(hparams)  # create job file (***.sh)
            job.run()
            y = job.collect_result()
            job.create_result_json(_create_result(hparams, float(y), trial_id))
            return float(y)

    def submit_parallel(self, objective, hparams_list, trial_ids) -> list[float]:
        results = []
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            future_to_job = {executor.submit(self.submit, objective, hparams, trial_id): trial_id for hparams, trial_id in zip(hparams_list, trial_ids)}
            for future in as_completed(future_to_job):
                trial_id = future_to_job[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    print(f'Trial {trial_id} generated an exception: {exc}')
        return results

    def wait(self) -> None:
        """Wait for the running jobs to finish."""
        ...

    def collect_results(self) -> list[dict[str, Any]]:
        """Collect the results of the finished jobs.

        return:
            list[tuple[float, int]]: The objective values and trial IDs of the finished jobs.
            [{"trial_id": 1", "params": {"x": 1.0, "y": 0.3}, "objective_value": 0.3}, ...]
        """
        ...

    ...

def _submit(self):
    """Submit the job."""
    ...

def _submit_parallel(self):
    """Submit the job in parallel."""
    ...


def _create_result(hparams: dict, y: float, trial_id: int) -> dict:
    """Create a result dictionary."""
    result = {"trial_id": trial_id, "objective_value": y}
    result.update(hparams)
    return result


def _run_objective(objective: Callable, params: dict) -> Any:
    """Run the objective function."""
    return objective(params)


"""

``` bash
python ex01.py
```

- ex01.py
``` python

import aiaccel
import optuna

from aiaccel import NelderMeadSampler


def objective(hparams: dict) -> float:
    x = hparams["x"]
    ...

    return (x - 2) ** 2


if __name__ == "__main__":

    study = optuna.create_study(
        direction='minimize',
        sampler=NelderMeadSampler(...)
    )

    jobs = aiaccel.JobDispatcher()
    n_trial = 100

    # Run the optimization loop
    for _ in range(n_trial):
        trial = study.ask()
        hparams = {
            'job_id': trial._trial_id,
            'x': trial.suggest_float('x', 0, 10),
        }
        trial, y = jobs.submit(objective, hparams, _tag_=trial)
        study.tell(trial, y)

    # ===================================================
    # Run the optimization loop with parallel execution
    # ===================================================
    # while True:
    #     for _ in range(jobs.availavle_n_jobs):
    #         trial = jobs.ask()
    #         hparams = {
    #             'x': trial.suggest_float('x', 0, 10),
    #             'job_id': 0
    #         }
    #         jobs.submit(objective, hparams, _tag_=trial)

    #     jobs.wait()

    #     for y, trial in jobs.collect_results():
    #         jobs.tell(trial, y)

```
"""

def objective(hparams: dict) -> float:
    x = hparams["x"]
    ...

    return (x - 2) ** 2

# ===================================================
# Run the optimization loop with parallel execution
# ===================================================
jobs = JobDispatcher()
while True:
    for _ in range(jobs.availavle_n_jobs):
        trial = jobs.ask()
        hparams = {
            'x': trial.suggest_float('x', 0, 10),
            'job_id': 0
        }
        jobs.submit(objective, hparams, _tag_=trial)

    jobs.wait()

    for y, trial in jobs.collect_results():
        jobs.tell(trial, y)
