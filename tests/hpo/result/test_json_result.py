from collections.abc import Generator
import json
import os
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

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
        source_dir = Path(__file__).parent
        test_files = ["config_for_json_test.yaml", "objective_for_json_test.sh", "objective_for_json_test.py"]

        for file_name in test_files:
            source_file = source_dir / file_name
            target_file = temp_path / file_name
            if source_file.exists():
                shutil.copy2(source_file, target_file)
                os.chmod(target_file, 0o755)
                print(f"\n=== Content of {file_name} ===")
                print((target_file).read_text())
                print("=" * 40)
            else:
                pytest.skip(f"Required test file {file_name} not found in {source_dir}")

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


def test_result(temp_dir: Path) -> None:
    with patch("sys.argv", ["optimize.py", "objective_for_json_test.sh", "--config", "config_for_json_test.yaml"]):
        from aiaccel.hpo.apps.optimize import main

        main()
        assert (temp_dir / "objective_for_json_test.sh_29_result.json").exists()
        assert (temp_dir / "objective_for_json_test.sh_30_result.json").exists() is False
