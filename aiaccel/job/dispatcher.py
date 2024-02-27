from __future__ import annotations

import copy
import sys
import time
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable
import uuid
from aiaccel.job.job_creator import JobCreator
from aiaccel.job.functions import param_to_args_key_value
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
        param_to_args_fn: Callable = param_to_args_key_value,
        retry_num: int = __DEFAULT_RETRY_NUM__,
        timeout_seconds: int = __DEFAULT_TOUT_SEC__,
        work_dir: str = __DEFAULT_WORK_DIR__,
    ):
        self.base_job_file_path = str(Path(base_job_file_path).resolve())
        self.platform = platform.lower()
        self.group = group
        self._n_jobs = n_jobs
        self.param_to_args_fn = param_to_args_fn
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
        hparams: dict,
        tag: Any = None,
        job_name: int | None = None,
    ) -> None:
        """Submit a job to the job dispatcher."""
        self._submit_job_count += 1
        job_name = job_name if job_name is not None else _get_job_name()
        hparams_str = self.param_to_args_fn(hparams)
        future = self.executor.submit(
            _create_and_run,
            self.base_job_file_path,
            job_name,
            self.platform,
            self.group,
            self.timeout_seconds,
            self.work_dir,
            hparams,
            hparams_str,
        )
        self.futures.append((future, job_name, hparams, tag))
        self._all_future.append(future)

        # Wait for at least one available worker
        while True:
            if self.available_worker_count > 0:
                break
            time.sleep(0.01)

    def _update_working_feature_list(self):
        self.futures = [f for f in self.futures if not f[0].done()]

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
            _, job_name, hparams, tag = next(
                (f, tid, hps, t) for f, tid, hps, t in self.futures if f == future
            )
            collected_results.append((result, tag))
            result = {"job_name": job_name, "value": result, "hparams": hparams}
            self._all_results.append(result)
            print(result)

        self._update_working_feature_list()
        return collected_results

    def result(self) -> Any:
        """Get the result of the job dispatcher."""
        future = self.futures.pop(0)  # get the first finished job
        y = future[0].result()  # wait for the completion of the job
        result = {"job_name": future[1], "value": y, "hparams": future[2]}
        self._all_results.append(result)
        print(result)
        self._update_working_feature_list()
        return y

    def results(self) -> list[dict]:
        return [f for f in self.futures if f[0].done()]

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
    hparams: dict,
    hparams_str: str,
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
    job.run(hparams_str)
    y = job.collect_result()
    job.create_result_json(_create_result(job_name, hparams, float(y)))
    return param_str_eval(y)


def _create_result(job_name: int, hparams: dict, y: float) -> dict:
    """Create a result dictionary."""
    result = {"job_name": job_name, "velue": y}
    result.update(hparams)
    return result
