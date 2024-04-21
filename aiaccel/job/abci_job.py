from __future__ import annotations

import re
import subprocess
import time
from enum import IntEnum, auto
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


class JobStatus(IntEnum):
    UNSUBMITTED = auto()
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()
    ERROR = auto()

    @classmethod
    def from_qsub(cls, status: str) -> JobStatus:
        match status:
            case "r":
                return JobStatus.RUNNING
            case "qw":
                return JobStatus.WAITING
            case "d":
                return JobStatus.RUNNING
            case "E":
                return JobStatus.ERROR
            case _:
                raise ValueError(f"Unexpected status: {status}")


class AbciJob:
    job_filename: Path
    job_group: str

    job_name: str

    cwd: Path
    stdout_filename: Path
    stderr_filename: Path

    tag: Any

    status: JobStatus
    job_number: int | None

    def __init__(
        self,
        job_filename: Path | str,
        job_group: str,
        job_name: str | None = None,
        cwd: Path | str | None = None,
        stdout_filename: Path | str | None = None,
        stderr_filename: Path | str | None = None,
        qsub_args: list[str] | None = None,
        args: list[str] | None = None,
        tag: Any = None,
    ):
        self.job_filename = Path(job_filename)
        self.job_group = job_group
        self.job_name = job_name if job_name is not None else self.job_filename.name

        self.cwd = Path(cwd) if cwd is not None else Path.cwd()
        self.stdout_filename = Path(stdout_filename) if stdout_filename is not None else self.cwd / f"{job_name}.o"
        self.stderr_filename = Path(stderr_filename) if stderr_filename is not None else self.cwd / f"{job_name}.o"

        self.tag = tag

        self.status = JobStatus.UNSUBMITTED
        self.job_number = None

        # generate qsub command
        self.cmd = ["qsub", "-g", job_group, "-o", str(self.stdout_filename), "-e", str(self.stderr_filename)]
        if self.job_name is not None:
            self.cmd += ["-N", self.job_name]
        if qsub_args is not None:
            self.cmd += [arg.format(job=self) for arg in qsub_args]
        self.cmd += [str(self.job_filename)]
        if args is not None:
            self.cmd += [arg.format(job=self) for arg in args]

    def submit(self) -> AbciJob:
        if self.status >= JobStatus.WAITING:
            raise RuntimeError(f"This job is already submited as {self.job_name} (id: {self.job_number})")

        p = subprocess.run(self.cmd, capture_output=True, text=True, check=True)

        match = re.search(r"Your job (\d+)", p.stdout)
        if match is None:
            raise RuntimeError(f"The following qsub result cannot be parsed: {p.stdout}")

        self.job_number = int(match.group(1))
        self.status = JobStatus.WAITING

        return self

    def update_status(self) -> JobStatus:
        self.update_status_batch([self])
        return self.status

    def wait(self, sleep_time: float = 10.0) -> AbciJob:
        while self.update_status() < JobStatus.FINISHED:
            time.sleep(sleep_time)

        return self

    @classmethod
    def update_status_batch(cls, job_list: list[AbciJob]) -> None:
        job_dict = {j.job_number: j for j in job_list if j.status not in [JobStatus.UNSUBMITTED, JobStatus.FINISHED]}
        p = subprocess.run(["qstat", "-xml"], capture_output=True, text=True, check=True)

        status_dict: dict[int, str] = {}
        for el in ElementTree.fromstring(p.stdout).iter("job_list"):
            status_dict[int(el.findtext("JB_job_number", default=-1))] = el.findtext("state", "")

        for job_number in set(job_dict.keys()) - set(status_dict.keys()):
            job_dict[job_number].status = JobStatus.FINISHED

        for job_number, status in status_dict.items():
            job_dict[job_number].status = JobStatus.from_qsub(status)
