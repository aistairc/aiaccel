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
    """
    Creates a mock JobFuture object with the given status.

    Parameters:
        status (JobStatus): The status of the job.

    Returns:
        JobFuture: The mock JobFuture object.
    """
    mock_future = MagicMock(spec=Future)
    mock_future.done.return_value = status in [JobStatus.FINISHED, JobStatus.ERROR]
    mock_future.running.return_value = status == JobStatus.RUNNING
    job_future = JobFuture(mock_future)
    job_future.status = status
    return job_future


def test_available_slots_full(executor: LocalJobExecutor) -> None:
    """
    Test case to verify the behavior of the `available_slots` method when all slots are full.

    Args:
        executor (LocalJobExecutor): The instance of the LocalJobExecutor class.

    Returns:
        None

    - Job status:
        - RUNNING: 4
    - available: 0
    """
    executor.job_list = [create_mock_job_future(JobStatus.RUNNING) for _ in range(4)]
    assert executor.available_slots() == 0


def test_available_slots_pending(executor: LocalJobExecutor) -> None:
    """
    Test case to verify the behavior of the `available_slots` method when there are pending jobs.

    Args:
        executor (LocalJobExecutor): An instance of the LocalJobExecutor class.

    Returns:
        None

    - Job status:
        - RUNNING: 2
    - available: 2
    """
    executor.job_list = [create_mock_job_future(JobStatus.RUNNING) for _ in range(2)]
    assert executor.available_slots() == 2


def test_available_slots_all_free(executor: LocalJobExecutor) -> None:
    """
    Test case to verify the available slots when all slots are free.

    Args:
        executor (LocalJobExecutor): The instance of the LocalJobExecutor class.

    Returns:
        None
    """
    assert executor.available_slots() == 4


def test_collect_finished(executor: LocalJobExecutor) -> None:
    """
    Test the `collect_finished` method of the `LocalJobExecutor` class.

    Args:
        executor (LocalJobExecutor): An instance of the `LocalJobExecutor` class.

    Returns:
        None

    - Job status:
        - RUNNING: 2
    - available: 0
    """
    executor.job_list = [
        create_mock_job_future(JobStatus.RUNNING),
        create_mock_job_future(JobStatus.RUNNING),
        create_mock_job_future(JobStatus.FINISHED),
        create_mock_job_future(JobStatus.FINISHED),
    ]

    result = executor.collect_finished()
    assert len(result) == 2


def test_collect_finished_empty(executor: LocalJobExecutor) -> None:
    """
    Test case to verify the behavior of the `collect_finished` method when the `job_list` is empty.

    Args:
        executor (LocalJobExecutor): An instance of the LocalJobExecutor class.

    Returns:
        None

    Raises:
        AssertionError: If the result of `collect_finished` is not an empty list.
        AssertionError: If the `job_list` is not empty after calling `collect_finished`.
    """

    executor.job_list = []

    result = executor.collect_finished()
    assert result == []
    assert executor.job_list == []


def test_update_status_batch(executor: LocalJobExecutor) -> None:
    """
    Test case for the `update_status_batch` method of the `LocalJobExecutor` class.

    This test verifies that the `update_status_batch` method correctly updates the status of all job futures
    in the `job_list` attribute of the `LocalJobExecutor` instance.

    Steps:
    1. Create a list of `JobFuture` objects with different statuses.
    2. Set the `job_list` attribute of the `executor` instance to the created list.
    3. Assert that the length of the `job_list` is 4.
    4. Assert that the status of each job future in the `job_list` is as expected.
    5. Call the `update_status_batch` method of the `executor` instance.
    6. Assert that the status of each job future in the `job_list` is updated to `JobStatus.WAITING`.

    This test ensures that the `update_status_batch` method correctly updates the status of all job futures
    in the `job_list` attribute of the `LocalJobExecutor` instance.
    """
    # Test code goes here
    pass
    job_list: list[JobFuture] = []

    # Create job futures with different statuses
    job1 = JobFuture(Future())
    job1.status = JobStatus.WAITING
    job_list.append(job1)

    job2 = JobFuture(Future())
    job2.status = JobStatus.RUNNING
    job_list.append(job2)

    job3 = JobFuture(Future())
    job3.status = JobStatus.FINISHED
    job_list.append(job3)

    job4 = JobFuture(Future())
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
