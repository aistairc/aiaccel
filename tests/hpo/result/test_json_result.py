from collections.abc import Generator
import json
import os
from pathlib import Path
import tempfile

import pytest

from aiaccel.hpo.job_executors import LocalJobExecutor
from aiaccel.hpo.results.json_result import JsonResult


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
    dst_filename = temp_dir / "result.json"
    with open(dst_filename, "w") as f:
        f.write(json.dumps({"objective": 42}))
    job = LocalJobExecutor(Path(""), work_dir=temp_dir)
    json_result = JsonResult("{job.cwd}/result.json")
    assert json_result.load(job) == 42


def test_load_float(temp_dir: Path) -> None:
    dst_filename = temp_dir / "result.json"
    with open(dst_filename, "w") as f:
        f.write(json.dumps({"objective": 3.14}))
    job = LocalJobExecutor(Path(""), work_dir=temp_dir)
    json_result = JsonResult("{job.cwd}/result.json")
    assert json_result.load(job) == 3.14


def test_load_str(temp_dir: Path) -> None:
    dst_filename = temp_dir / "result.json"
    with open(dst_filename, "w") as f:
        f.write(json.dumps({"objective": "result"}))
    job = LocalJobExecutor(Path(""), work_dir=temp_dir)
    json_result = JsonResult("{job.cwd}/result.json")
    assert json_result.load(job) == "result"
