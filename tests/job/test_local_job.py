import shutil
from collections.abc import Generator
from pathlib import Path

import pytest

from aiaccel.job import LocalJob
from aiaccel.job.local_job import JobStatus


@pytest.fixture
def job_instance(tmpdir: Path) -> Generator[LocalJob, None, None]:
    cwd = tmpdir / "cwd"
    cwd.mkdir()

    job_name = "job"

    yield LocalJob(
        job_filename="job.sh",
        job_name=job_name,
        cwd=str(cwd),
        args=["arg1", "arg2"],
        tag=None,
    )

    shutil.rmtree(str(cwd))


def test_init(job_instance: LocalJob) -> None:
    job = job_instance
    assert isinstance(job.job_filename, Path)
    assert isinstance(job.cwd, Path)
    assert job.status == JobStatus.UNSUBMITTED
    assert job.cmd == (
        ["bash"]
        + ["job.sh", "arg1", "arg2"]
    )


def test_submit(job_instance: LocalJob) -> None:
    job = job_instance

    assert job.status == JobStatus.UNSUBMITTED

    result = job.submit()

    assert result == job
    assert result.status == JobStatus.RUNNING


def test_update_status(job_instance: LocalJob) -> None:
    job = job_instance

    assert job.status == JobStatus.UNSUBMITTED

    result = job.submit()
    result.update_status()

    assert result.status == JobStatus.RUNNING
