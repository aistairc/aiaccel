# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from pathlib import Path
import subprocess

import pytest

from tests.job.apps.base_job_tests import BaseJobTests


class TestLocal(BaseJobTests):
    cmd = ["aiaccel-job", "local"]
    config_test_path = Path(__file__).parent / "config" / "custom_local.yaml"

    # Skip test_cpu_cancel_job in local
    def test_cpu_cancel_job(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        pass

    def run_cpu_array(self, log_path: Path) -> None:
        subprocess.run(
            self.make_command()
            + [
                "cpu",
                "--n_tasks",
                "2",
                "--n_tasks_per_proc",
                "1",
                "--n_procs",
                "2",
                log_path,
                "--",
                "bash",
                "-c",
                'echo "TASK_INDEX=$TASK_INDEX"',
            ],
            check=True,
        )

    def assert_cpu_array(self, tmp_path: Path, log_path: Path) -> None:
        log_prefix = log_path.with_suffix("")
        for process_index in [1, 2]:
            log_file = tmp_path / f"{log_prefix.name}.{process_index}.log"
            assert log_file.exists()
            assert log_file.read_text().strip() == f"TASK_INDEX={process_index}"

    def run_cpu_fails_when_log_write_fails(self, log_path: Path) -> None:
        subprocess.run(
            self.make_command() + ["cpu", log_path, "--", "echo", "job succeeded"],
            check=True,
        )

    def assert_cpu_fails_when_log_write_fails(
        self, exc_info: pytest.ExceptionInfo[subprocess.CalledProcessError], status_path: Path
    ) -> None:
        assert exc_info.value.returncode == 1
