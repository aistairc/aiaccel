import shutil
from collections.abc import Generator
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aiaccel.job import LocalJobExecutor
from aiaccel.job.job_status import JobStatus
from aiaccel.job.local_job_executor import JobFuture


@pytest.fixture
def executor(tmpdir: Path) -> Generator[LocalJobExecutor, None, None]:
    work_dir = tmpdir / "workdir"
    work_dir.mkdir()

    n_max_jobs = 4

    yield LocalJobExecutor(
        job_filename="main.sh",
        job_name="job",
        work_dir=str(work_dir),
        n_max_jobs=n_max_jobs,
    )

    shutil.rmtree(str(work_dir))


def create_mock_job_future(status: JobStatus) -> JobFuture:
    mock_future = MagicMock(spec=Future)
    mock_future.done.return_value = status in [JobStatus.FINISHED, JobStatus.ERROR]
    mock_future.running.return_value = status == JobStatus.RUNNING
    job_future = JobFuture(mock_future)
    job_future.status = status
    return job_future


def test_available_slots_full(executor: LocalJobExecutor) -> None:
    executor.job_list = [create_mock_job_future(JobStatus.RUNNING) for _ in range(4)]
    assert executor.available_slots() == 0


def test_available_slots_pending(executor: LocalJobExecutor) -> None:
    executor.job_list = [create_mock_job_future(JobStatus.RUNNING) for _ in range(2)]
    assert executor.available_slots() == 2


def test_available_slots_empty(executor: LocalJobExecutor) -> None:
    assert executor.available_slots() == 4


def test_collect_finished(executor: LocalJobExecutor) -> None:
    executor.job_list = [
        create_mock_job_future(JobStatus.RUNNING),
        create_mock_job_future(JobStatus.RUNNING),
        create_mock_job_future(JobStatus.FINISHED),
        create_mock_job_future(JobStatus.FINISHED),
    ]

    result = executor.collect_finished()
    assert len(result) == 2


def test_collect_finished_empty(executor: LocalJobExecutor) -> None:
    executor.job_list = []

    result = executor.collect_finished()
    assert result == []
    assert executor.job_list == []
