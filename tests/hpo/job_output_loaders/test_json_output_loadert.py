from collections.abc import Generator
import json
import os
from pathlib import Path
import shutil
import tempfile
from unittest.mock import patch

from omegaconf import OmegaConf as oc  # noqa: N813

import pytest

from aiaccel.hpo.job_executors import LocalJobExecutor
from aiaccel.hpo.job_output_loaders.json_loader import JsonJobOutputLoader


@pytest.fixture
def temp_dir() -> Generator[Path]:
    """Create a temporary directory with test files and clean up afterwards"""

    # create a config file
    config_path = os.path.join(os.path.dirname(__file__), "config_base.yaml")
    base = oc.load(config_path)
    output_loadeer = {
        "result": {
            "_target_": "aiaccel.hpo.job_output_loaders.JsonJobOutputLoader",
            "filename_template": "{job.cwd}/{job.job_name}_result.json",
        }
    }
    config = oc.merge(base, output_loadeer)

    # create objective script
    src = """
#!/bin/bash
python objective_for_output_loaders.py $@ "--output_type" "json"
"""

    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_path = Path(tmp_dir)
        original_dir = os.getcwd()
        os.chdir(tmp_dir)
        source_dir = Path(__file__).parent

        source_file = source_dir / "objective_for_output_loaders.py"
        target_file = temp_path / "objective_for_output_loaders.py"
        shutil.copy2(source_file, target_file)
        os.chmod(target_file, 0o755)

        # generate config file
        oc.save(config, temp_path / "config_for_json_test.yaml")

        # generate objective script
        with open(temp_path / "objective_for_json_test.sh", "w") as f:
            f.write(src)

        yield temp_path
        os.chdir(original_dir)


def test_load_int(temp_dir: Path) -> None:
    dst_filename = temp_dir / "result.json"
    with open(dst_filename, "w") as f:
        f.write(json.dumps(42))
    loader = JsonJobOutputLoader("{job.cwd}/result.json")
    job_executor = LocalJobExecutor(Path(""), loader=loader, work_dir=temp_dir)
    job = job_executor.submit([])
    json_result = JsonJobOutputLoader("{job.cwd}/result.json")
    assert json_result.load(job) == 42


def test_load_float(temp_dir: Path) -> None:
    dst_filename = temp_dir / "result.json"
    with open(dst_filename, "w") as f:
        f.write(json.dumps(3.14))

    laoder = JsonJobOutputLoader("{job.cwd}/result.json")
    job_executor = LocalJobExecutor(Path(""), loader=laoder, work_dir=temp_dir)
    job = job_executor.submit([])
    json_result = JsonJobOutputLoader("{job.cwd}/result.json")
    assert json_result.load(job) == 3.14


def test_load_str(temp_dir: Path) -> None:
    dst_filename = temp_dir / "result.json"
    with open(dst_filename, "w") as f:
        f.write(json.dumps("result"))
    loader = JsonJobOutputLoader("{job.cwd}/result.json")
    executor = LocalJobExecutor(Path(""), loader=loader, work_dir=temp_dir)
    job = executor.submit([])
    json_result = JsonJobOutputLoader("{job.cwd}/result.json")
    assert json_result.load(job) == "result"


def test_result(temp_dir: Path) -> None:
    with patch("sys.argv", ["optimize.py", "objective_for_json_test.sh", "--config", "config_for_json_test.yaml"]):
        from aiaccel.hpo.apps.optimize import main

        main()
        assert (temp_dir / "objective_for_json_test.sh_29_result.json").exists()
        assert (temp_dir / "objective_for_json_test.sh_30_result.json").exists() is False
