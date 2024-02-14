from __future__ import annotations
import copy
import sys
import time
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable
import uuid
from aiaccel.job.job_creator import JobCreator
import ast

__default_work_dir__ = "./work"
__default_timeout_seconds__ = -1  # no timeout
__default_retry_num__ = 0  # no retry
__default_n_jobs__ = 1  # no parallel execution

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


def _eval(s: str) -> Any:
    try:
        return ast.literal_eval(s)
    except ValueError:
        return s


class JobDispatcher:
    def __init__(
        self,
        func: Callable,
        n_trials: int,
        platform: str = "",
        group: str = "",
        preamble: str = "",
        n_jobs: int = __default_n_jobs__,
        retry_num: int = __default_retry_num__,
        timeout_seconds: int = __default_timeout_seconds__,
        work_dir: str = __default_work_dir__,
    ):
        self.func = func
        if args.e:
            hparams = {}
            for k, v in hp_args.items():
                hparams[k] = _eval(v)
            _run_job(self.func, hparams)
            sys.exit(0)
        # ====
        self.n_trials = n_trials
        self.platform = platform.lower()
        self.group = group
        self._n_jobs = n_jobs
        self.preamble = preamble
        self.retry_num = retry_num
        self.timeout_seconds = timeout_seconds
        self.work_dir = Path(work_dir).resolve()
        self.futures = []
        self._all_future = []
        self._submit_job_count = 0
        self.all_results = []
        self.script_name = sys.argv[0]

        if not self.work_dir.exists():
            self.work_dir.mkdir(parents=True)

        self.executor = ProcessPoolExecutor(max_workers=n_jobs)

    def submit(
        self, hparams: dict, tag: Any = None, job_name: int | None = None
    ) -> None:
        """Submit a job to the job dispatcher."""
        self._submit_job_count += 1
        job_name = job_name if job_name is not None else self.get_job_name()

        future = self.executor.submit(
            _create_and_run,
            self.script_name,
            job_name,
            self.platform,
            self.group,
            self.preamble,
            self.timeout_seconds,
            self.work_dir,
            hparams,
        )
        self.futures.append((future, job_name, hparams, tag))
        self._all_future.append(future)

        # Wait for at least one available worker
        while True:
            if self.available_worker_count > 0:
                break
            if self.all_done():
                break
            time.sleep(0.1)

    def _update_working_feature_list(self):
        self.futures = [f for f in self.futures if not f[0].done()]

    def collect_results(self) -> list[tuple[float, Any]]:
        """Collect the results of the finished jobs.

        return:
            List of tuples containing the objective value and the corresponding trial object.
        """
        fdone = [f for f in self.futures if f[0].done()]
        if len(fdone) == 0:
            return []

        collected_results = []
        for future in [f for f, _, _, _ in fdone]:
            result = future.result()
            _, job_name, hparams, tag = next(
                (f, tid, hps, t) for f, tid, hps, t in self.futures if f == future
            )
            collected_results.append((result, tag))
            self.all_results.append(
                {"job_name": job_name, "value": result, "hparams": hparams}
            )

        self._update_working_feature_list()
        return collected_results

    @property
    def available_worker_count(self) -> int:
        _working_feature_count = len([f for f in self.futures if not f[0].done()])
        return min(
            self.n_trials - self.submit_job_count, self._n_jobs - _working_feature_count
        )

    @property
    def finished_job_count(self) -> int:
        """Get the number of finished jobs."""
        if len(self._all_future) == 0:
            return 0
        return len([f for f in self._all_future if f.done()])

    @property
    def submit_job_count(self) -> int:
        """Get the number of submitted jobs."""
        return self._submit_job_count

    def all_done(self) -> bool:
        if len(self._all_future) >= self.n_trials:
            return all([f.done() for f in self._all_future])
        else:
            return False

    def get_job_name(self) -> int:
        return str(uuid.uuid4())

    @property
    def results(self) -> list[dict]:
        all_results = copy.deepcopy(self.all_results)
        all_results.sort(key=lambda x: x["job_name"])
        return all_results

    ...


def _run_job(objective: Callable, hparams: dict):
    # Called by the job script file (***.sh), not invoked by the job dispatcher.
    # Retrieve hyperparameter values from command-line arguments and update the parameter object.
    # Execute the objective function and print the result.
    y = _run_objective(objective, hparams)
    sys.stdout.write(f"{str(y)}\n")
    sys.stdout.flush()


def _create_and_run(
    script_name: str,
    job_name: int,
    platform: str,
    group: str,
    preamble: str,
    timeout_seconds: int,
    work_dir: Path,
    hparams: dict,
):
    job = JobCreator(
        script_name,
        job_name,
        platform,
        group,
        preamble,
        timeout_seconds,
        work_dir,
    )
    job.create(hparams)  # create job file (***.sh)
    job.run()
    y = job.collect_result()
    job.create_result_json(_create_result(job_name, hparams, float(y)))
    return _eval(y)


def _run_objective(objective: Callable, params: dict) -> Any:
    """Run the objective function."""
    return objective(params)


def _create_result(job_name: int, hparams: dict, y: float) -> dict:
    """Create a result dictionary."""
    result = {"job_name": job_name, "velue": y}
    result.update(hparams)
    return result
