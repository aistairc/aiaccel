from collections.abc import Generator
from pathlib import Path
import re
import shutil
from unittest.mock import patch

import pytest
from utils import qstat_xml

from aiaccel.hpo.job import AbciJob
from aiaccel.hpo.job.jobs.job_status import JobStatus

#

## AbciJob


@pytest.fixture
def job_instance(tmpdir: Path) -> Generator[AbciJob, None, None]:
    cwd = tmpdir / "cwd"
    cwd.mkdir()

    job_name = "job"

    yield AbciJob(
        job_filename=Path("./job.sh"),
        job_group="group",
        job_name=job_name,
        cwd=str(cwd),
        stdout_filename=cwd / f"{job_name}.o",
        stderr_filename=cwd / f"{job_name}.e",
        qsub_args=["-l"],
        args=["arg1", "arg2"],
        tag=None,
    )

    shutil.rmtree(str(cwd))


def test_init(job_instance: AbciJob) -> None:
    job = job_instance
    assert isinstance(job.job_filename, Path)
    assert isinstance(job.cwd, Path)
    assert isinstance(job.stdout_filename, Path)
    assert isinstance(job.stderr_filename, Path)
    assert job.status == JobStatus.UNSUBMITTED
    assert job.job_number is None
    assert job.cmd == (
        ["qsub"]
        + ["-g", job.job_group]
        + ["-o", str(job.stdout_filename)]
        + ["-e", str(job.stderr_filename)]
        + ["-N", job.job_name]
        + ["-l", str(job.job_filename)]
        + ["arg1", "arg2"]
    )


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

    job_list: list[AbciJob] = []
    for ii in range(4):
        job = AbciJob(job_filename=Path("job.sh"), job_group="group", job_name="job")
        job.status = JobStatus.WAITING
        job.job_number = 42340793 + 2 * ii

        job_list.append(job)

    with patch("subprocess.run", return_value=qstat_xml("tests/job/qstat_dat.txt")):
        AbciJob.update_status_batch(job_list)

    for job in job_list:
        assert job.status == JobStatus.RUNNING
