from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import Future
from pathlib import Path
from typing import Any
import uuid
from aiaccel.job.job_creator import JobCreator
from aiaccel.job.eval import param_str_eval


__DEFAULT_WORK_DIR__ = "./work"
__DEFAULT_TOUT_SEC__ = -1  # no timeout
__DEFAULT_RETRY_NUM__ = 0  # no retry
__DEFAULT_N_JOBS__ = 1  # no parallel execution


class JobDispatcher:
    def __init__(
        self,
        base_job_file_path: str,
        platform: str = "",
        group: str = "",
        n_jobs: int = __DEFAULT_N_JOBS__,
        retry_num: int = __DEFAULT_RETRY_NUM__,
        timeout_seconds: int = __DEFAULT_TOUT_SEC__,
        work_dir: str = __DEFAULT_WORK_DIR__,
    ):
        self.base_job_file_path = str(Path(base_job_file_path).resolve())
        self.platform = platform.lower()
        self.group = group
        self._n_jobs = n_jobs
        self.retry_num = retry_num  # not used yet
        self.timeout_seconds = timeout_seconds  # not used yet
        self.work_dir = Path(work_dir).resolve()

        self.futures = []
        self._all_future = []
        self._submit_job_count = 0
        self._all_results = []

        if not self.work_dir.exists():
            self.work_dir.mkdir(parents=True)

        self.executor = ProcessPoolExecutor(max_workers=n_jobs)

    def submit(
        self,
        args: list,
        tag: Any = None,
        job_name: int | None = None,
    ) -> Future:
        """Submit a job to the job dispatcher."""
        self._submit_job_count += 1
        job_name = job_name if job_name is not None else _get_job_name()
        future = self.executor.submit(
            _create_and_run,
            self.base_job_file_path,
            job_name,
            self.platform,
            self.group,
            self.timeout_seconds,
            self.work_dir,
            args,
        )
        self.futures.append((future, job_name, args, tag))
        self._all_future.append(future)

        # Wait for at least one available worker
        while True:
            if self.available_worker_count > 0:
                break
            time.sleep(0.01)

        return future

    def wait(self):
        """Wait for all jobs to finish."""
        for future in self._all_future:
            future.result()

    ########################################
    # collect result
    ########################################

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
            result = future.result()  # wait for the completion of the job
            _, job_name, args, tag = next(
                (f, tid, hps, t) for f, tid, hps, t in self.futures if f == future
            )
            collected_results.append((result, tag))
            result = {"job_name": job_name, "value": result, "args": args}
            self._all_results.append(result)

        self._update_working_feature_list()
        return collected_results

    def _update_working_feature_list(self):
        self.futures = [f for f in self.futures if not f[0].done()]

    ########################################
    # status
    ########################################

    @property
    def available_worker_count(self) -> int:
        _working_feature_count = len([f for f in self.futures if not f[0].done()])
        return self._n_jobs - _working_feature_count

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

    ...


def _get_job_name() -> int:
    return str(uuid.uuid4())


def _create_and_run(
    base_job_file_path: str,
    job_name: int,
    platform: str,
    group: str,
    timeout_seconds: int,
    work_dir: Path,
    args: list,
):
    job = JobCreator(
        base_job_file_path,
        job_name,
        platform,
        group,
        timeout_seconds,
        work_dir,
    )
    job.create()
    job.run(args)
    y = job.collect_result()
    job.create_result_json(_create_result(job_name, args, float(y)))
    return param_str_eval(y)


def _create_result(job_name: int, args: list, y: float) -> dict:
    """Create a result dictionary."""
    result = {"job_name": job_name, "velue": y}
    result.update({"args": args})
    return result
