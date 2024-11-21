from collections.abc import Generator
from concurrent.futures import Future
from pathlib import Path
import shutil
from unittest.mock import MagicMock

import pytest

from aiaccel.job import LocalJobExecutor
from aiaccel.job.job_status import JobStatus
from aiaccel.job.local_job import LocalJob


@pytest.fixture
def executor(tmpdir: Path) -> Generator[LocalJobExecutor, None, None]:
    work_dir = tmpdir / "workdir"
    work_dir.mkdir()

    n_max_jobs = 4

    yield LocalJobExecutor(
        job_filename=Path("main.sh"),
        job_name="job",
        work_dir=str(work_dir),
        n_max_jobs=n_max_jobs,
    )

    shutil.rmtree(str(work_dir))


def create_mock_job_future(status: JobStatus) -> LocalJob:
    mock_future = MagicMock(spec=Future)
    mock_future.done.return_value = status in [JobStatus.FINISHED, JobStatus.ERROR]
    mock_future.running.return_value = status == JobStatus.RUNNING
    job_future = LocalJob(mock_future, Path("main.sh"))
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


def test_update_status_batch(executor: LocalJobExecutor) -> None:
    job_list: list[LocalJob] = []

    # Create job futures with different statuses
    job1 = LocalJob(Future(), Path("main.sh"))
    job1.status = JobStatus.WAITING
    job_list.append(job1)

    job2 = LocalJob(Future(), Path("main.sh"))
    job2.status = JobStatus.RUNNING
    job_list.append(job2)

    job3 = LocalJob(Future(), Path("main.sh"))
    job3.status = JobStatus.FINISHED
    job_list.append(job3)

    job4 = LocalJob(Future(), Path("main.sh"))
    job4.status = JobStatus.ERROR
    job_list.append(job4)

    executor.job_list = job_list

    assert len(executor.job_list) == 4
    assert executor.job_list[0].status == JobStatus.WAITING  # type: ignore
    assert executor.job_list[1].status == JobStatus.RUNNING  # type: ignore
    assert executor.job_list[2].status == JobStatus.FINISHED  # type: ignore
    assert executor.job_list[3].status == JobStatus.ERROR  # type: ignore

    # Update the status batch
    executor.update_status_batch()

    assert executor.job_list[0].status == JobStatus.WAITING  # type: ignore
    assert executor.job_list[1].status == JobStatus.WAITING  # type: ignore
    assert executor.job_list[2].status == JobStatus.WAITING  # type: ignore
    assert executor.job_list[3].status == JobStatus.WAITING  # type: ignore
