import shutil
from collections.abc import Generator
from concurrent.futures import Future
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aiaccel.job import LocalJobExecutor


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

    executor.job_list = [MagicMock(spec=Future) for _ in range(4)]

    assert executor.available_slots() == 0


def test_available_slots_pending(executor: LocalJobExecutor) -> None:
    """
    - Job status:
        - RUNNING: 2
    - available: 2
    """

    executor.job_list = [MagicMock(spec=Future) for _ in range(2)]

    assert executor.available_slots() == 2


def test_available_slots_empty(executor: LocalJobExecutor) -> None:
    """
    - Job status:
        - RUNNING: 0
        - WAITING: 0
    - available: 4
    """

    assert executor.available_slots() == 4


def test_collect_finished(executor: LocalJobExecutor) -> None:
    """
    - Job status:
        - RUNNING: 2
        - WAITING: 1
        - FINISHED: 1
    - finished: 1
    - still_running: 2
    """

    executor.job_list = [
        MagicMock(spec=Future, done=lambda: False),
        MagicMock(spec=Future, done=lambda: False),
        MagicMock(spec=Future, done=lambda: True),
        MagicMock(spec=Future, done=lambda: True),
    ]

    result = executor.collect_finished()
    assert len(result) == 2


def test_collect_finished_empty(executor: LocalJobExecutor) -> None:
    executor.job_list = []

    result = executor.collect_finished()
    assert result == []
    assert executor.job_list == []
