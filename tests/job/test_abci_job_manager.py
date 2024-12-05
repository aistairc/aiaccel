from collections.abc import Generator
from pathlib import Path
import shutil
from unittest.mock import MagicMock, patch

import pytest
from utils import qstat_xml

from aiaccel.hpo.job_executors import AbciJobExecutor
from aiaccel.hpo.jobs import AbciJob
from aiaccel.hpo.jobs.job_status import JobStatus


@pytest.fixture
def executor(tmpdir: Path) -> Generator[AbciJobExecutor, None, None]:
    work_dir = tmpdir / "workdir"
    work_dir.mkdir()

    n_max_jobs = 4

    yield AbciJobExecutor(
        job_filename=Path("main.sh"),
        job_group="group",
        job_name="job",
        work_dir=str(work_dir),
        n_max_jobs=n_max_jobs,
    )

    shutil.rmtree(str(work_dir))


def test_available_slots_full(executor: AbciJobExecutor) -> None:
    """
    - Job status:
        - RUNNING: 4
        - available: 0
    """

    for ii in range(4):
        job = AbciJob(executor.job_filename, executor.job_group, job_name=executor.job_name)
        job.status = JobStatus.RUNNING
        job.job_number = 42340793 + 2 * ii

        executor.job_list.append(job)

    with patch("subprocess.run", return_value=qstat_xml("tests/job/qstat_dat.txt")):
        slots = executor.available_slots()
        assert slots == executor.n_max_jobs - len([j for j in executor.job_list if j.status <= JobStatus.RUNNING])


def test_available_slots_pending(executor: AbciJobExecutor) -> None:
    """
    - Job status:
        - RUNNING: 2
        - WAITING: 1
    - available: 1
    """

    for ii, status in enumerate([JobStatus.RUNNING, JobStatus.RUNNING, JobStatus.WAITING]):
        job = AbciJob(executor.job_filename, executor.job_group, job_name=executor.job_name)
        job.status = status
        job.job_number = 42340793 + 2 * ii

        executor.job_list.append(job)

    with patch("subprocess.run", return_value=qstat_xml("tests/job/qstat_dat 2r_1qw.txt")):
        slots = executor.available_slots()
        assert slots == 1


def test_available_slots_full_running(executor: AbciJobExecutor) -> None:
    """
    - Job status:
        - RUNNING: 4
        - WAITING: 0
    - available: 0
    """

    for ii in range(4):
        job = AbciJob(executor.job_filename, executor.job_group, job_name=executor.job_name)
        job.status = JobStatus.RUNNING
        job.job_number = 42340793 + 2 * ii

        executor.job_list.append(job)

    with patch("subprocess.run", return_value=qstat_xml("tests/job/qstat_dat.txt")):
        result = executor.available_slots()
        assert result == 0


def test_available_slots_empty(executor: AbciJobExecutor) -> None:
    """
    - Job status:
        - RUNNING: 0
        - WAITING: 0
    - available: 4
    """
    with patch("subprocess.run", return_value=qstat_xml("tests/job/qstat_dat_empty.txt")):
        result = executor.available_slots()
        assert result == 4


def test_collect_finished(executor: AbciJobExecutor) -> None:
    job_list: list[AbciJob] = []
    for status in [JobStatus.FINISHED, JobStatus.RUNNING, JobStatus.FINISHED]:
        job = MagicMock(spec=AbciJob)
        job.status = status

        job_list.append(job)
        executor.job_list.append(job)

    result = executor.collect_finished()
    assert result == [job_list[0], job_list[2]]
    assert executor.job_list == [job_list[1]]


def test_collect_finished_empty(executor: AbciJobExecutor) -> None:
    executor.job_list = []

    result = executor.collect_finished()
    assert result == []
    assert executor.job_list == []
