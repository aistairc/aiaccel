import shutil
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aiaccel.job import LocalJob, LocalJobExecutor
from aiaccel.job.local_job import JobStatus


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


def test_available_slots_full(executor: LocalJobExecutor) -> None:
    """
    - Job status:
        - RUNNING: 4
        - available: 0
    """

    for _ in range(4):
        job = LocalJob(executor.job_filename, job_name=executor.job_name)
        job.submit()
        job.status = JobStatus.RUNNING
        executor.job_list.append(job)

    assert executor.available_slots() == executor.n_max_jobs - len(
        [j for j in executor.job_list if j.status <= JobStatus.RUNNING])



def test_available_slots_pending(executor: LocalJobExecutor) -> None:
    """
    - Job status:
        - RUNNING: 2
        - UNSUBMITTED: 1
    - available: 1
    """

    for _, status in enumerate([JobStatus.RUNNING, JobStatus.RUNNING, JobStatus.UNSUBMITTED]):
        job = LocalJob(executor.job_filename, job_name=executor.job_name)
        job.submit()
        job.status = status
        executor.job_list.append(job)

    assert executor.available_slots() == 1


def test_available_slots_empty(executor: LocalJobExecutor) -> None:
    """
    - Job status:
        - RUNNING: 0
        - WAITING: 0
    - available: 4
    """

    assert executor.available_slots() == 4


def test_collect_finished(executor: LocalJobExecutor) -> None:
    job_list: list[LocalJob] = []
    for status in [JobStatus.FINISHED, JobStatus.RUNNING, JobStatus.FINISHED]:
        job = MagicMock(spec=LocalJob)
        job.status = status

        job_list.append(job)
        executor.job_list.append(job)

    result = executor.collect_finished()
    assert result == [job_list[0], job_list[2]]
    assert executor.job_list == [job_list[1]]


def test_collect_finished_empty(executor: LocalJobExecutor) -> None:
    executor.job_list = []

    result = executor.collect_finished()
    assert result == []
    assert executor.job_list == []
