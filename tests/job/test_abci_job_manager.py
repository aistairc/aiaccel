import shutil
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aiaccel.job import AbciJob, AbciJobExecutor, JobStatus


class SubprocessReturn:
    stdout = ""
    stderr = ""


def qstat_xml(txt_data_path: str = "tests/job/qstat_dat.txt") -> SubprocessReturn:
    p = SubprocessReturn()
    with open(txt_data_path) as f:
        p.stdout = f.read()
    return p


@pytest.fixture
def executor(tmpdir: Path) -> Generator[AbciJobExecutor, None, None]:
    job_filename = "main.sh"
    job_group = "group"
    job_name = "job"
    work_dir = tmpdir / "workdir"
    work_dir.mkdir()
    n_max_jobs = 4
    executor = AbciJobExecutor(
        job_filename,
        job_group,
        job_name=job_name,
        work_dir=str(work_dir),
        n_max_jobs=n_max_jobs,
    )
    yield executor
    shutil.rmtree(str(work_dir))


def test_available_slots_full(executor: AbciJobExecutor) -> None:
    """
    - Job status:
        - RUNNING: 4
        - available: 0
    """
    job1 = AbciJob(executor.job_filename, executor.job_group, job_name=executor.job_name)
    job1.status = JobStatus.RUNNING
    job1.job_number = 42340793
    job2 = AbciJob(executor.job_filename, executor.job_group, job_name=executor.job_name)
    job2.status = JobStatus.RUNNING
    job2.job_number = 42340795
    job3 = AbciJob(executor.job_filename, executor.job_group, job_name=executor.job_name)
    job3.status = JobStatus.RUNNING
    job3.job_number = 42340797
    job4 = AbciJob(executor.job_filename, executor.job_group, job_name=executor.job_name)
    job4.status = JobStatus.RUNNING
    job4.job_number = 42340799
    executor.job_list = [job1, job2, job3, job4]

    with patch("subprocess.run", return_value=qstat_xml("tests/job/qstat_dat.txt")):
        result = executor.available_slots()
        num_available_slots = executor.n_max_jobs - len(
            [j for j in executor.job_list if j.status == JobStatus.RUNNING or JobStatus.WAITING]
        )
        assert result == num_available_slots


def test_available_slots_pending(executor: AbciJobExecutor) -> None:
    """
    - Job status:
        - RUNNING: 2
        - WAITING: 1
    - available: 1
    """
    job1 = AbciJob(executor.job_filename, executor.job_group, job_name=executor.job_name)
    job1.status = JobStatus.RUNNING
    job1.job_number = 42340793
    job2 = AbciJob(executor.job_filename, executor.job_group, job_name=executor.job_name)
    job2.status = JobStatus.RUNNING
    job2.job_number = 42340795
    job3 = AbciJob(executor.job_filename, executor.job_group, job_name=executor.job_name)
    job3.status = JobStatus.WAITING
    job3.job_number = 42340797
    executor.job_list = [job1, job2, job3]

    with patch("subprocess.run", return_value=qstat_xml("tests/job/qstat_dat 2r_1qw.txt")):
        result = executor.available_slots()
        assert result == 1


def test_available_slots_full_running(executor: AbciJobExecutor) -> None:
    """
    - Job status:
        - RUNNING: 4
        - WAITING: 0
    - available: 0
    """
    executor.job_list = []

    job1 = AbciJob(executor.job_filename, executor.job_group, job_name=executor.job_name)
    job1.status = JobStatus.RUNNING
    job1.job_number = 42340793
    job2 = AbciJob(executor.job_filename, executor.job_group, job_name=executor.job_name)
    job2.status = JobStatus.RUNNING
    job2.job_number = 42340795
    job3 = AbciJob(executor.job_filename, executor.job_group, job_name=executor.job_name)
    job3.status = JobStatus.RUNNING
    job3.job_number = 42340797
    job4 = AbciJob(executor.job_filename, executor.job_group, job_name=executor.job_name)
    job4.status = JobStatus.RUNNING
    job4.job_number = 42340799
    executor.job_list = [job1, job2, job3, job4]

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
    executor.job_list = []

    with patch("subprocess.run", return_value=qstat_xml("tests/job/qstat_dat_empty.txt")):
        result = executor.available_slots()
        assert result == 4


def test_collect_finished(executor: AbciJobExecutor) -> None:
    job1 = MagicMock(spec=AbciJob)
    job1.status = JobStatus.FINISHED
    job2 = MagicMock(spec=AbciJob)
    job2.status = JobStatus.RUNNING
    job3 = MagicMock(spec=AbciJob)
    job3.status = JobStatus.FINISHED
    executor.job_list = [job1, job2, job3]

    result = executor.collect_finished()
    assert result == [job1, job3]
    assert job1 not in executor.job_list
    assert job3 not in executor.job_list


def test_collect_finished_empty(executor: AbciJobExecutor) -> None:
    executor.job_list = []

    result = executor.collect_finished()
    assert result == []
    assert executor.job_list == []
