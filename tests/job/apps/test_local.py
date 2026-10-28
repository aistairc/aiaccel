# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import os
from pathlib import Path
import subprocess

import pytest

from tests.job.apps.job_test_base import JobTestBase


class TestLocal(JobTestBase):
    cmd = ["aiaccel-job", "local"]
    config_test_path = Path(__file__).parent / "config" / "custom_local.yaml"

    def test_cpu_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        log_path = tmp_path / "test.log"

        result = subprocess.run(
            self.make_command()
            + [
                "cpu",
                log_path,
                "--",
                "bash",
                "-c",
                "echo before failure; exit 7",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert log_path.exists()
        assert "before failure\n" in log_path.read_text()

    def test_local_fails_when_tee_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)

        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        fake_tee = fake_bin / "tee"
        # Make tee fail deterministically after consuming the job output.
        fake_tee.write_text("#!/bin/sh\ncat >/dev/null\nexit 1\n")
        fake_tee.chmod(0o755)

        # Keep aiaccel-job and other system commands available.
        current_path = os.environ.get("PATH", "")
        monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{current_path}")

        log_path = tmp_path / "test.log"

        # The job succeeds; only tee should make the invocation fail.
        with pytest.raises(subprocess.CalledProcessError):
            subprocess.run(
                self.make_command() + ["cpu", log_path, "--", "echo", "job succeeded"],
                check=True,
            )

    def test_cpu_array(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        log_path = tmp_path / "test_log.log"

        subprocess.run(
            self.make_command()
            + [
                "cpu",
                "--n_tasks",
                "2",
                "--n_tasks_per_proc",
                "1",
                "--n_procs",
                "1",
                log_path,
                "--",
                "bash",
                "-c",
                'echo "TASK_INDEX=$TASK_INDEX"',
            ],
            check=True,
        )

        # Local runs one process for each process slot; it does not create
        # one scheduler-style log per task index.
        log_file = tmp_path / "test_log.1.log"
        assert log_file.exists()
        assert log_file.read_text().strip() == "TASK_INDEX=1"
