from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Generator
import uuid
from aiaccel.job.job_creator import JobCreator
from aiaccel.job.eval import param_str_eval
from dataclasses import dataclass



@dataclass
class AbciJob:
    future: Future
    job_name: str
    args: list
    tag: Any

    def is_finished(self) -> bool:
        return self.future.done()

    def retrieve_result(self) -> Any:
        return self.future.result()

    def get_result(self, interval: float = 1.0) -> Any:
        while not self.is_finished():
            time.sleep(interval)
        return self.retrieve_result()

    def cancel(self) -> None:
        self.future.cancel()


class AbciJobExecutor:
    def __init__(
        self,
        base_job_file_path: str,
        group: str = "",
        n_jobs: int = 1,
        retry_num: int = 0,
        timeout_seconds: float = -1.0,
        work_dir: str = "./work",
    ):
        self.base_job_file_path = str(Path(base_job_file_path).resolve())
        self.group: str = group
        self._n_jobs: int = n_jobs
        self.retry_num: int = retry_num  # not used yet
        self.timeout_seconds: float = timeout_seconds  # not used yet
        self.work_dir: Path = Path(work_dir).resolve()

        self._submit_job_count: int = 0
        self._finished_job_count: int = 0
        self.job_list: list[AbciJob] = []  # for reference
        self._active_job_list: list[AbciJob] = []

        if not self.work_dir.exists():
            self.work_dir.mkdir(parents=True)

        self.executor = ProcessPoolExecutor(max_workers=n_jobs)

    def _submit(self, args: list, tag: Any, job_name: int | None) -> AbciJob:
        self._submit_job_count += 1
        job_name = job_name if job_name is not None else _get_job_name()
        future = self.executor.submit(
            _create_and_run,
            self.base_job_file_path,
            job_name,
            self.group,
            self.timeout_seconds,
            self.work_dir,
            args,
        )
        return AbciJob(future, job_name, args, tag)

    def submit(
        self, args: list, tag: Any = None, job_name: int | None = None
    ) -> AbciJob:
        job = self._submit(args, tag=tag, job_name=job_name)
        self.job_list.append(job)
        self._active_job_list.append(job)

        # Wait for at least one available worker
        while True:
            if self.available_worker_count > 0:
                break
            time.sleep(0.01)

        return job

    def get_results(self) -> Generator:
        finished_jobs = [job for job in self._active_job_list if job.future.done()]
        for job in finished_jobs:
            self._active_job_list.pop(self._active_job_list.index(job))
            self._finished_job_count += 1
            yield job.get_result(), job.tag

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True)

    @property
    def available_worker_count(self) -> int:
        _working_feature_count = len(
            [f.future for f in self.job_list if not f.future.done()]
        )
        return self._n_jobs - _working_feature_count

    @property
    def finished_job_count(self) -> int:
        return self._finished_job_count

    @property
    def submit_job_count(self) -> int:
        return self._submit_job_count

    ...



def _get_job_name() -> int:
    return str(uuid.uuid4())


def _create_and_run(
    base_job_file_path: str,
    job_name: int,
    group: str,
    timeout_seconds: int,
    work_dir: Path,
    args: list,
):
    job_file = JobCreator(
        base_job_file_path,
        job_name,
        group,
        timeout_seconds,
        work_dir,
    )
    job_file.create()
    job_file.run(args)
    y = job_file.collect_result()
    result = {"job_name": job_name, "velue": y}
    result.update({"args": args})
    job_file.create_result_json(result)
    return param_str_eval(y)
