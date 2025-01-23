from collections.abc import Generator
import pickle as pkl

import os
from pathlib import Path
import tempfile

import pytest

from aiaccel.hpo.job_executors import LocalJobExecutor
from aiaccel.hpo.results.pickle_result import PickleResult


@pytest.fixture
def temp_dir() -> Generator[Path]:
    """Create a temporary directory with test files and clean up afterwards"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        original_dir = os.getcwd()
        os.chdir(tmp_dir)
        yield temp_path
        os.chdir(original_dir)


def test_load_int(temp_dir: Path) -> None:
    dst_filename = temp_dir / "result.pkl"
    with open(dst_filename, "wb") as f:
        pkl.dump(42, f)

    job = LocalJobExecutor(Path(""), work_dir=temp_dir)
    pkl_result = PickleResult("{job.cwd}/result.pkl")
    assert pkl_result.load(job) == 42


def test_load_float(temp_dir: Path) -> None:
    dst_filename = temp_dir / "result.pkl"
    with open(dst_filename, "wb") as f:
        pkl.dump(3.14, f)

    job = LocalJobExecutor(Path(""), work_dir=temp_dir)
    pkl_result = PickleResult("{job.cwd}/result.pkl")
    assert pkl_result.load(job) == 3.14


def test_load_str(temp_dir: Path) -> None:
    dst_filename = temp_dir / "result.pkl"
    with open(dst_filename, "wb") as f:
        pkl.dump("result", f)

    job = LocalJobExecutor(Path(""), work_dir=temp_dir)
    pkl_result = PickleResult("{job.cwd}/result.pkl")
    assert pkl_result.load(job) == "result"
