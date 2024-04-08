from __future__ import annotations

import re
import subprocess
import time
import uuid
from collections.abc import Generator
from enum import Enum
from pathlib import Path
from typing import Any

import xmltodict


class JobStatus(Enum):
    UNSUBMITTED = 0
    WAITING = 1
    RUNNING = 2
    FINISHED = 3
    ERROR = 4


status_mapping = {
    "r": JobStatus.RUNNING,  # running
    "qw": JobStatus.WAITING,  # waiting
    "d": JobStatus.RUNNING,  # deleting
    "E": JobStatus.ERROR,  # Error
}


def qstat_xml() -> dict[Any, Any]:
    cmd = "qstat -xml"
    data = subprocess.run(cmd.split(), capture_output=True, text=True)
    qstat_dict = xmltodict.parse(data.stdout)
    return qstat_dict


class AbciJob:
    def __init__(
        self,
        job_name: str,
        args: list[Any],
        tag: Any,
        job_file_path: Path,
        stdout_file_path: Path,
        stderr_file_path: Path,
    ) -> None:
        self.job_name = job_name
        self.args = args
        self.tag = tag
        self.job_file_path = job_file_path
        self.stdout_file_path = stdout_file_path
        self.stderr_file_path = stderr_file_path
        self.job_number: int | None = None
        self.status: JobStatus = JobStatus.UNSUBMITTED

    def run(self, cmd: str) -> None:
        print(f"{cmd}")

        output = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        pattern = r"Your job (\d+)"
        match = re.search(pattern, output.stdout)

        if match:
            self.job_number = int(match.group(1))
        else:
            raise ValueError(f"Failed to submit the job: {self.job_name}")

        for _ in range(10):
            self.update_state()
            if self.status != JobStatus.UNSUBMITTED:
                break
            time.sleep(3)
            # wait for the job to be submitted

        if self.status == JobStatus.UNSUBMITTED:
            raise ValueError(f"Failed to submit the job: {self.job_name}")

    def _get_job_status(self, job_number: int) -> str | None:
        """
        Get the status of the job.
        """
        qstat = qstat_xml()
        if qstat["job_info"]["queue_info"] is None:
            return None

        job_list = qstat["job_info"]["queue_info"]["job_list"]
        if isinstance(job_list, dict):
            if int(job_list["JB_job_number"]) == job_number:
                return str(job_list["state"])
            return None

        for job_list in qstat["job_info"]["queue_info"]["job_list"]:
            if int(job_list["JB_job_number"]) == job_number:
                return str(job_list["state"])
        return None

    def update_state(self) -> None:
        if self.job_number is None:
            raise ValueError("Job number is not set.")
        status = self._get_job_status(self.job_number)
        if self.status != JobStatus.UNSUBMITTED and status is None:
            self.status = JobStatus.FINISHED
        else:
            if status is not None:
                self.status = status_mapping[status]

    def set_state(self, state: JobStatus) -> None:
        self.status = state

    def collect_result(self, retry_left: int = 10, wait: float = 1.0) -> str:
        """Collect the result of the job.

        return:
            The result of the job (objective value).

        Note:
            Retry reading the file if it is not found.
            This is because the file metadata may not be updated immediately after the job finishes.
        """
        while retry_left > 0:
            try:
                with open(str(self.stdout_file_path), encoding="utf-8") as file:
                    lines = file.readlines()
                    if lines:
                        return lines[-1].strip()
                    else:
                        raise ValueError("No result found.")

            except FileNotFoundError:
                time.sleep(wait)
                retry_left -= 1

            except ValueError:
                time.sleep(wait)
                retry_left -= 1

        raise ValueError("No result found.")

    def get_result(self) -> Any:
        y = self.collect_result()
        self.result = y
        return self.result


class AbciJobExecutor:
    working_job_list: list[AbciJob] = []

    def __init__(
        self,
        job_file_path: str,
        n_jobs: int = 1,
        work_dir: Path | str = "./work",
    ):
        self.job_file_path = Path(job_file_path).resolve()
        self._n_jobs: int = n_jobs
        self.work_dir: Path = Path(work_dir).resolve()
        if not self.work_dir.exists():
            self.work_dir.mkdir(parents=True)
        self._submit_job_count: int = 0
        self._finished_job_count: int = 0

    def submit(self, args: list[str], group: str, job_name: str = "", tag: Any | None = None) -> AbciJob:
        if job_name == "":
            job_name = str(uuid.uuid4())
        stdout_file_path = self.work_dir / f"{job_name}.o"
        stderr_file_path = self.work_dir / f"{job_name}.e"
        args_str = " ".join(args)
        qsub_cmd = (
            f"qsub -g {group} -N {job_name} -o {stdout_file_path} -e {stderr_file_path} {self.job_file_path} {args_str}"
        )
        return self._execute(qsub_cmd, job_name, args, tag, stdout_file_path, stderr_file_path)

    def _execute(
        self,
        cmd: str,
        job_name: str,
        args: list[str],
        tag: Any | None,
        stdout_file_path: Path,
        stderr_file_path: Path,
    ) -> AbciJob:
        self._submit_job_count += 1

        job = AbciJob(
            job_name,
            args,
            tag,
            self.job_file_path,
            stdout_file_path,
            stderr_file_path,
        )

        job.run(cmd)
        self.working_job_list.append(job)

        # Wait for at least one available worker
        while True:
            self.update_state_batch()
            if self.available_worker_count > 0:
                break
            time.sleep(3)
        return job

    def get_results(self) -> Generator[tuple[Any, Any], None, None]:
        self.update_state_batch()
        for job in self.working_job_list:
            if job.status == JobStatus.FINISHED:
                self.working_job_list.pop(self.working_job_list.index(job))
                self._finished_job_count += 1
                yield job.get_result(), job.tag

    def get_result(self) -> Any:
        self.update_state_batch()
        if self.working_job_list[-1].status == JobStatus.FINISHED:
            job = self.working_job_list.pop(0)
            self._finished_job_count += 1
            return job.get_result()
        else:
            return None

    @classmethod
    def update_state_batch(cls) -> None:
        def _update_state(job_number: int, state: str) -> None:
            for job in cls.working_job_list:
                if job.job_number == job_number:
                    job.set_state(status_mapping[state])
                    break
                job.set_state(JobStatus.FINISHED)

        qstat = qstat_xml()
        queue_info = qstat["job_info"].get("queue_info", None)
        if queue_info is not None:
            job_list = queue_info.get("job_list", None)
            if job_list is not None:
                if isinstance(job_list, dict):
                    _update_state(int(job_list["JB_job_number"]), job_list["state"])
                else:
                    for job in job_list:
                        _update_state(int(job["JB_job_number"]), job["state"])

        job_info = qstat["job_info"].get("job_info", None)
        if job_info is not None:
            job_list = job_info.get("job_list", None)
            if job_list is not None:
                if isinstance(job_list, dict):
                    _update_state(int(job_list["JB_job_number"]), job_list["state"])
                else:
                    for job in job_list:
                        _update_state(int(job["JB_job_number"]), job["state"])
        else:
            for job in cls.working_job_list:
                if job.status != JobStatus.UNSUBMITTED:
                    job.set_state(JobStatus.FINISHED)

    @property
    def available_worker_count(self) -> int:
        _working_job_count = len([j for j in self.working_job_list if j.status != JobStatus.FINISHED])
        return self._n_jobs - _working_job_count

    @property
    def finished_job_count(self) -> int:
        return self._finished_job_count

    @property
    def submit_job_count(self) -> int:
        return self._submit_job_count

    @property
    def job_list(self) -> list[AbciJob]:
        return self.working_job_list
