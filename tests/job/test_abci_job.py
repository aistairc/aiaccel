import re
import shutil
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest

from aiaccel.job import AbciJob, JobStatus


class SubprocessReturn:
    stdout = ""
    stderr = ""


def qstat_xml(txt_data_path: str = "tests/job/qstat_dat.txt") -> SubprocessReturn:
    p = SubprocessReturn()
    with open(txt_data_path) as f:
        p.stdout = f.read()
    return p


## JobStatus


def test_from_qsub_running() -> None:
    assert JobStatus.from_qsub("r") == JobStatus.RUNNING
    assert JobStatus.from_qsub("d") == JobStatus.RUNNING
    assert JobStatus.from_qsub("Rr") == JobStatus.RUNNING


def test_from_qsub_waiting() -> None:
    assert JobStatus.from_qsub("qw") == JobStatus.WAITING
    assert JobStatus.from_qsub("h") == JobStatus.WAITING
    assert JobStatus.from_qsub("t") == JobStatus.WAITING
    assert JobStatus.from_qsub("s") == JobStatus.WAITING
    assert JobStatus.from_qsub("S") == JobStatus.WAITING
    assert JobStatus.from_qsub("T") == JobStatus.WAITING
    assert JobStatus.from_qsub("Rq") == JobStatus.WAITING


def test_from_qsub_error() -> None:
    assert JobStatus.from_qsub("E") == JobStatus.ERROR


def test_from_qsub_unexpected_status() -> None:
    with pytest.raises(ValueError, match="Unexpected status: unexpected"):
        JobStatus.from_qsub("unexpected")


## AbciJob


@pytest.fixture
def job_instance(tmpdir: Path) -> Generator[AbciJob, None, None]:
    job_filename: str = "job.sh"
    job_group: str = "group"
    job_name: str = "job"
    cwd: Path = tmpdir / "cwd"
    cwd.mkdir()
    stdout_filename: Path = Path(cwd) / f"{job_name}.o"
    stderr_filename: Path = Path(cwd) / f"{job_name}.e"
    qsub_args: list[str] = ["-l"]
    args: list[str] = ["arg1", "arg2"]
    tag: None = None

    job: AbciJob = AbciJob(
        job_filename,
        job_group,
        job_name=job_name,
        cwd=str(cwd),
        stdout_filename=stdout_filename,
        stderr_filename=stderr_filename,
        qsub_args=qsub_args,
        args=args,
        tag=tag,
    )

    yield job

    shutil.rmtree(str(cwd))


def test_init(job_instance: AbciJob) -> None:
    job = job_instance
    assert isinstance(job.job_filename, Path)
    assert isinstance(job.cwd, Path)
    assert isinstance(job.stdout_filename, Path)
    assert isinstance(job.stderr_filename, Path)
    assert job.status == JobStatus.UNSUBMITTED
    assert job.job_number is None
    assert job.cmd == [
        "qsub",
        "-g",
        job.job_group,
        "-o",
        str(job.stdout_filename),
        "-e",
        str(job.stderr_filename),
        "-N",
        job.job_name,
        "-l",
        str(job.job_filename),
        "arg1",
        "arg2",
    ]


def test_submit(job_instance: AbciJob) -> None:
    job = job_instance
    job_number = 123456

    assert job.status == JobStatus.UNSUBMITTED

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = "Your job 123456"
        result = job.submit()

    assert result == job
    assert result.job_number == job_number
    assert result.status == JobStatus.WAITING
    mock_run.assert_called_once_with(job.cmd, capture_output=True, text=True, check=True)


def test_submit_already_submitted(job_instance: AbciJob) -> None:
    job = job_instance
    job.job_number = 123456
    error_message = f"This job is already submited as {job.job_name} (id: {job.job_number})"

    with pytest.raises(RuntimeError, match=re.escape(error_message)):
        job.status = JobStatus.WAITING
        job.submit()


def test_submit_qsub_result_cannot_be_parsed(job_instance: AbciJob) -> None:
    job = job_instance
    error_message = "The following qsub result cannot be parsed: Invalid qsub result"

    assert job.status == JobStatus.UNSUBMITTED

    with patch("subprocess.run") as mock_run:
        mock_run.return_value.stdout = "Invalid qsub result"

        with pytest.raises(RuntimeError, match=re.escape(error_message)):
            job.submit()


def test_update_status(job_instance: AbciJob) -> None:
    job = job_instance
    job.job_number = 42340793

    with patch("subprocess.run", return_value=qstat_xml("tests/job/qstat_dat_1r.txt")):
        job.status = JobStatus.WAITING
        result = job.update_status()

    assert result == JobStatus.RUNNING


def test_wait(job_instance: AbciJob) -> None:
    job = job_instance

    with patch.object(job, "update_status") as mock_update_status:
        mock_update_status.side_effect = [
            JobStatus.RUNNING,
            JobStatus.RUNNING,
            JobStatus.FINISHED,
        ]
        result = job.wait(sleep_time=0.1)

    assert result == job
    assert mock_update_status.call_count == 3
    mock_update_status.assert_called_with()


def test_update_status_batch() -> None:
    """
    - Job status:
        - WAITING: 4
        - RUNNING: 0

    ↓↓↓↓↓

    - Job status:
        - WAITING: 0
        - RUNNING: 4
    """

    job_filename = "job.sh"
    job_group = "group"
    job_name = "job"

    job1 = AbciJob(job_filename, job_group, job_name=job_name)
    job1.status = JobStatus.WAITING
    job1.job_number = 42340793

    job2 = AbciJob(job_filename, job_group, job_name=job_name)
    job2.status = JobStatus.WAITING
    job2.job_number = 42340795

    job3 = AbciJob(job_filename, job_group, job_name=job_name)
    job3.status = JobStatus.WAITING
    job3.job_number = 42340797

    job4 = AbciJob(job_filename, job_group, job_name=job_name)
    job4.status = JobStatus.WAITING
    job4.job_number = 42340799

    job_list = [job1, job2, job3, job4]

    with patch("subprocess.run", return_value=qstat_xml("tests/job/qstat_dat.txt")):
        AbciJob.update_status_batch(job_list)

    assert job1.status == JobStatus.RUNNING
    assert job2.status == JobStatus.RUNNING
    assert job3.status == JobStatus.RUNNING
    assert job4.status == JobStatus.RUNNING
