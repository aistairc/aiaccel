import os
from pathlib import Path
import subprocess
import tempfile


def test_local() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        log_path = str(Path(tmp_dir) / "test.log")
        # default
        command = ["aiaccel-job", "local", "cpu", log_path]
        command += ["--", "ls"]

        subprocess.run(command, check=True)

        # load config from args
        config_path = str(Path(__file__).parent / "job_config.yaml")
        command = ["aiaccel-job", "local", "--config", config_path, "cpu", log_path]
        command += ["--", "ls"]

        subprocess.run(command, check=True)

        # load config from ENV
        os.environ["AIACCEL_JOB_CONFIG"] = str(Path(__file__).parent / "job_config.yaml")
        command = ["aiaccel-job", "local", "cpu", log_path]
        command += ["--", "ls"]

        subprocess.run(command, check=True)
