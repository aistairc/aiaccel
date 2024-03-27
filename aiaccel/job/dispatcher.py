from __future__ import annotations

import ast
import subprocess
import time
import uuid

import xmltodict

from pathlib import Path
from typing import Any, Generator

from aiaccel.job.retry import retry


class AbciJob:
    def __init__(
        self,
        cmd: str,
        job_name: str,
        args: list[Any],
        tag: Any,
        job_file_path: Path,
        stdout_file_path: Path,
        stderr_file_path: Path,
    ) -> None:
        self.cmd = cmd
        self.job_name = job_name
        self.args = args
        self.tag = tag
        self.job_file_path = job_file_path
        self.stdout_file_path = stdout_file_path
        self.stderr_file_path = stderr_file_path
        self._job_number: int | None = None
        self._submitted: bool = False
        self._seen_job_numbers: set[int] = set()
        self._state: str | None = None

    def is_finished(self) -> bool:
        """
        投入済み かつ qstat で 自job_number の state が取得できなければ 終了と判定
        """
        self.update_state()
        if self._submitted and self.get_state() is None:
            return True
        return False

    def run(self) -> None:
        self._cleanup_output_files()
        self._execute_command()
        self._update_job_number()
        self.update_state()
        if self.get_state() is not None and self.get_state().lower() == "e":
            raise ValueError(f"Failed to submit the job: {self.job_name}")
        self._submitted = True

    def _get_job_status(self, job_number: int | None) -> str | None:
        """
        Get the status of the job.
        """
        if job_number is None:
            return None
        qstat = qstat_xml()
        if qstat["job_info"]["queue_info"] is None:
            return None

        job_list = qstat["job_info"]["queue_info"]["job_list"]
        if isinstance(job_list, dict):
            if int(job_list["JB_job_number"]) == job_number:
                return job_list["state"]
            return None

        for job_list in qstat["job_info"]["queue_info"]["job_list"]:
            if int(job_list["JB_job_number"]) == job_number:
                return job_list["state"]
        return None

    def update_state(self) -> None:
        if self._job_number is None:
            self._update_job_number()
        self._state = self._get_job_status(self._job_number)

    def get_state(self) -> str | None:
        return self._state

    def set_state(self, state: str) -> None:
        self._state = state

    @retry(_MAX_NUM=10, _DELAY=1.0)
    def collect_result(self) -> str:
        """Collect the result of the job.

        return:
            The result of the job (objective value).

        Note:
            Retry reading the file if it is not found.
            This is because the file metadata may not be updated immediately after the job finishes.
        """
        with open(str(self.stdout_file_path), encoding="utf-8") as file:
            lines = file.readlines()
            if lines:
                return lines[-1].strip()
            else:
                raise ValueError("No result found.")

    def get_result(self) -> Any:
        y = self.collect_result()
        try:
            y = ast.literal_eval(y)
        except ValueError:
            y = y
        self.result = y
        return self.result

    def get_job_numbers(self, job_name: str) -> list[int]:
        """
        Get the job numbers of the same job name.
        """
        qstat = qstat_xml()
        job_numbers = []

        if qstat["job_info"]["queue_info"] is None:
            return job_numbers

        job_list = qstat["job_info"]["queue_info"]["job_list"]
        if isinstance(job_list, dict):
            job_numbers.append(int(job_list["JB_job_number"]))
            return job_numbers

        for job in qstat["job_info"]["queue_info"]["job_list"]:
            if job["JB_name"] == job_name:
                job_numbers.append(int(job["JB_job_number"]))
        return job_numbers

    @retry(_MAX_NUM=10, _DELAY=3.0)
    def _update_job_number(self) -> None:
        jn = self.get_job_numbers(self.job_name)
        for job_number in jn:
            if job_number not in self._seen_job_numbers:
                self._job_number = job_number
                self._seen_job_numbers.add(job_number)
                break
        if self._job_number is None:
            raise ValueError(f"Job number not found: {self.job_name}")

    def _cleanup_output_files(self) -> None:
        if self.stdout_file_path.exists():
            self.stdout_file_path.unlink()
        if self.stderr_file_path.exists():
            self.stderr_file_path.unlink()

    def _execute_command(self) -> None:
        print(f"{self.cmd}")
        subprocess.run(self.cmd.split(), capture_output=True, text=True)

    def set_seen_job_number(self, job_number: int) -> None:
        self._seen_job_numbers.add(job_number)

    @property
    def job_number(self) -> int | None:
        return self._job_number


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
        self._create_work_dir()

        self._submit_job_count: int = 0
        self._finished_job_count: int = 0

    def _create_work_dir(self) -> None:
        if not self.work_dir.exists():
            self.work_dir.mkdir(parents=True)

    def submit(
        self, args: list[str], group: str, job_name: str = "", tag: Any | None = None
    ) -> AbciJob:
        if job_name == "":
            job_name = str(uuid.uuid4())
        stdout_file_path = self.work_dir / f"{job_name}.o"
        stderr_file_path = self.work_dir / f"{job_name}.e"
        args_str = " ".join(args)
        qsub_cmd = f"qsub -g {group} -N {job_name} -o {stdout_file_path} -e {stderr_file_path} {self.job_file_path} {args_str}"
        # 生成コマンド例:
        #  qsub -g [group] -N [job_name] -o [stdout_file_path] -e [stderr_file_path] [job_file_path] [--x0 ... --x1 ...]

        return self._execute(
            qsub_cmd, job_name, args, tag, stdout_file_path, stderr_file_path
        )

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
            cmd,
            job_name,
            args,
            tag,
            self.job_file_path,
            stdout_file_path,
            stderr_file_path,
        )

        for j in self.working_job_list:
            if j.job_number is not None:
                job.set_seen_job_number(j.job_number)

        job.run()
        self.working_job_list.append(job)

        # Wait for at least one available worker
        while True:
            self.update_state_batch()
            if self.available_worker_count > 0:
                break
            time.sleep(3)
        return job

    def get_results(self) -> Generator[tuple[Any, Any], None, None]:
        for job in self.working_job_list:
            if job.is_finished():
                self.working_job_list.pop(self.working_job_list.index(job))
                self._finished_job_count += 1
                yield job.get_result(), job.tag

    def get_result(self) -> Any:
        if self.working_job_list[-1].is_finished():
            job = self.working_job_list.pop(0)
            self._finished_job_count += 1
            return job.get_result()
        else:
            return None

    @classmethod
    def update_state_batch(cls) -> None:
        qstat = qstat_xml()
        qstat_job_list = qstat["job_info"]["queue_info"]["job_list"]
        if qstat_job_list is None:
            return

        qstat_states = {}
        if isinstance(qstat_job_list, dict):
            qstat_states[int(qstat_job_list["JB_job_number"])] = qstat_job_list["state"]
        else:
            for job in qstat_job_list:
                qstat_states[int(job["JB_job_number"])] = job["state"]

        for job in cls.working_job_list:
            if job.job_number in qstat_states.keys():
                job.set_state(qstat_states[job.job_number])
            else:
                job.set_state(None)

    @property
    def available_worker_count(self) -> int:
        _working_job_count = len(
            [j for j in self.working_job_list if not j.is_finished()]
        )
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

    ...


def qstat_xml() -> dict:
    cmd = "qstat -xml"
    data = subprocess.run(cmd.split(), capture_output=True, text=True)
    qstat_dict = xmltodict.parse(data.stdout)
    return qstat_dict
