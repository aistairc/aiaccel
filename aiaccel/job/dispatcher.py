from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Generator
import uuid
from aiaccel.job.eval import param_str_eval
from dataclasses import dataclass


from aiaccel.job.retry import retry
import fcntl

import subprocess
import json


@dataclass
class AbciJob:
    work_dir: InitVar[Path]
    future: Future
    args: list
    tag: Any

    def __post_init__(self, work_dir: Path):  # 定数類はABCIJobで管理すると良いかと
        self.job_name = str(uuid.uuid4())
        
        self.job_filename = work_dir / f"{self.job_name}.sh"
        self.stdout_filename = work_dir / f"{self.job_name}.o"
        self.stderr_filename = work_dir / f"{self.job_name}.e"

        self.lock_filename = work_dir / f"{self.job_name}.lock"
        self.result_filename = work_dir / f"{self.job_name}.json"

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
            job_name,
            self.group,
            args,
            self.base_job_file_path,
            self.get_job_file_path(job_name),
            self.get_lock_file_path(job_name),
            self.get_stdout_file_path(job_name),
            self.get_stderr_file_path(job_name),
            self.get_result_file_path(job_name),
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

    def get_job_file_path(self, job_name: str) -> Path:
        return self.work_dir / f"{job_name}.sh"

    def get_stdout_file_path(self, job_name: int) -> Path:
        return self.work_dir / f"{job_name}.o"

    def get_stderr_file_path(self, job_name: int) -> Path:
        return self.work_dir / f"{job_name}.e"

    def get_lock_file_path(self, job_name: int) -> Path:
        return self.work_dir / f"{job_name}.lock"

    def get_result_file_path(self, job_name: int) -> Path:
        return self.work_dir / f"{job_name}.json"

    ...



def _get_job_name() -> int:
    return str(uuid.uuid4())


def create_submit_command(
    group: str,
    stdout_file_path: str,
    stderr_file_path: str,
    job_file_path: str,
    args: list
) -> str:
    args_str = " ".join(args)
    return f"qsub -g {group} -o {stdout_file_path} -e {stderr_file_path} {job_file_path} {args_str}"


def create_job_file(
    base_job_file_path: str,
    job_file_path: str,
    lock_file_path: str,
) -> None:
    """Create a executable file to run the job."""
    with open(base_job_file_path, "r") as f:
        batch_file = f.read()

    with open(job_file_path, "w") as f:
        f.write(f"#!/bin/bash\n")
        f.write(f"LOCKFILE={lock_file_path}\n")
        f.write(f'if [ ! -f "$LOCKFILE" ]; then\n')
        f.write(f'  touch "$LOCKFILE"\n')
        f.write(f"fi\n")
        # lock
        f.write(f'flock -x -n "$LOCKFILE"\n')
        if batch_file:
            f.write(f"\n{batch_file}\n")
        # unlock
        f.write(f'flock -u "$LOCKFILE"\n')
        f.write(f'rm -f "$LOCKFILE"\n')


def run(
    group: str,
    job_file_path: str,
    stdout_file_path: str,
    stderr_file_path: str,
    lock_file_path: str,
    args: list
) -> None:
    """Run the job with the given hyperparameters."""

    def _wait_for_unlock() -> None:
        """Wait until the lock file is unlocked."""
        if not Path(lock_file_path).exists():
            return

        while True:
            with open(lock_file_path, "r") as f:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except IOError:
                    time.sleep(0.01)

    def _wait_for_lock_file_creation() -> None:
        """Wait until the lock file is created."""
        while True:
            if Path(lock_file_path).exists():
                break
            time.sleep(0.01)

    cmd = create_submit_command(
        group,
        stdout_file_path,
        stderr_file_path,
        job_file_path,
        args
    )

    print(f"Running the job with the command: `{cmd}`")
    cmds = cmd.split()
    subprocess.run(cmds, capture_output=True, text=True)  # is run in the another node
    _wait_for_lock_file_creation()
    _wait_for_unlock()


@retry(_MAX_NUM=60, _DELAY=1.0)
def collect_result(stdout_file_path: str) -> str | None:
    """Collect the result of the job.

    return:
        The result of the job (objective value).

    Note:
        Retry reading the file if it is not found.
        This is because the file metadata may not be updated immediately after the job finishes.
    """
    with open(stdout_file_path, encoding="utf-8") as file:
        lines = file.readlines()
        if lines:
            return lines[-1].strip()
        else:
            raise ValueError("No result found.")


def create_result_json(result_file_path, result: dict) -> None:
    """Create a json file to store the result of the job.
    The file name is `{job_name}.json`.
    """
    with open(result_file_path, "w") as f:
        json.dump(result, f)

def _create_and_run(
    job_name: int,
    group: str,
    args: list,
    base_job_file_path: str,
    job_file_path: str,
    lock_file_path: str,
    stdout_file_path: str,
    stderr_file_path: str,
    result_file_path: str,
):
    create_job_file(base_job_file_path, job_file_path, lock_file_path)
    run(
        group,
        job_file_path,
        stdout_file_path,
        stderr_file_path,
        lock_file_path,
        args
    )
    y = collect_result(stdout_file_path)
    result = {"job_name": job_name, "velue": y}
    result.update({"args": args})
    create_result_json(result_file_path, result)
    return param_str_eval(y)
