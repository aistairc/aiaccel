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
        platform: str = "",
        group: str = "",
        preamble: str = "",
        n_jobs: int = __default_n_jobs__,
        retry_num: int = __default_retry_num__,
        timeout_seconds: int = __default_timeout_seconds__,
        work_dir: str = __default_work_dir__,
    ):
        self.platform = platform.lower()
        self.group = group
        self._n_jobs = n_jobs
        self.preamble = preamble
        self.retry_num = retry_num
        self.timeout_seconds = timeout_seconds
        self.work_dir = Path(work_dir).resolve()
        self.executor = ThreadPoolExecutor(max_workers=n_jobs)
        self.futures = []
        self.results = []
        self._finished_job_count = 0
        self.all_result = []

        if not self.work_dir.exists():
            self.work_dir.mkdir(parents=True)

    @property
    def abvailable_worker_count(self) -> int:
        return self._n_jobs - len(self.futures)

    @property
    def finished_job_count(self) -> int:
        """Get the number of finished jobs."""
        return self._finished_job_count

    def _run(self, objective: Callable, hparams: dict, trial_id: int) -> float:
        """Run the job with the specified hyperparameters and collect the objective value."""
        if args.e:
            # Called by the job script file (***.sh), not invoked by the job dispatcher.
            # Retrieve hyperparameter values from command-line arguments and update the parameter object.
            # Execute the objective function and print the result.
            for k, v in hp_args.items():
                hparams[k] = float(v)
            y = _run_objective(objective, hparams)
            sys.stdout.write(f"{str(y)}\n")
            sys.stdout.flush()
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
            job.create_result_json(_create_result(trial_id, hparams, float(y)))
            return float(y)

    def submit(self, objective, hparams, trial_id, _tag_) -> None:
        """Submit a job to the job dispatcher."""
        future = self.executor.submit(self._run, objective, hparams, trial_id)
        self.futures.append((future, trial_id, hparams, _tag_))

    def wait(self) -> None:
        """Wait for the running jobs to finish."""
        futures = [f for f, _, _, _ in self.futures]
        for future in as_completed(futures):
            result = future.result()
            _, trial_id, hparams, _tag_ = next(
                (f, tid, hps, t) for f, tid, hps, t in self.futures if f == future
            )
            self.results.append((trial_id, hparams, result, _tag_))
            self._finished_job_count += 1
        self.futures = []

    def collect_results(self) -> list[tuple[float, Any]]:
        """Collect the results of the finished jobs.

        return:
            List of tuples containing the objective value and the corresponding trial object.
        """
        collected_results = []
        for trial_id, hparams, result, _tag_ in self.results:
            collected_results.append((result, _tag_))
            self.all_result.append(
                {"trial_id": trial_id, "hparams": hparams, "objective": result}
            )
        self.results = []  # Reset the results after collecting
        return collected_results

    ...


def _create_result(trial_id: int, hparams: dict, y: float) -> dict:
    """Create a result dictionary."""
    result = {"trial_id": trial_id, "objective": y}
    result.update(hparams)
    return result


def _run_objective(objective: Callable, params: dict) -> Any:
    """Run the objective function."""
    return objective(params)
