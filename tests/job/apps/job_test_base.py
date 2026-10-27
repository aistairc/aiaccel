# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from pathlib import Path
import subprocess

import pytest


class JobTestBase:
    cmd: list[str]
    config_path: Path | None = None
    config_test_path: Path

    def make_command(self) -> list[str | Path]:
        command: list[str | Path] = list(self.cmd.copy())

        if self.config_path is not None:
            command += ["--config", self.config_path]

        return command

    @pytest.mark.parametrize(
        "base_args",
        [
            ["cpu"],
            ["cpu", "--n_tasks=10"],
            ["gpu"],
            ["gpu", "--n_tasks=10"],
        ],
    )
    def test_default(self, base_args: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        log_path = tmp_path / "test.log"

        subprocess.run(
            self.make_command() + base_args + [log_path, "--", "sleep", "0"],
            check=True,
        )

    def test_config_from_argparse(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        log_path = tmp_path / "test.log"

        subprocess.run(
            self.cmd + ["--config", self.config_test_path, "cpu", log_path, "sleep", "0"],
            check=True,
        )

        with open(tmp_path / "config_path.txt") as f:
            assert Path(f.read().rstrip("\n")) == self.config_test_path

    def test_config_from_environ(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        log_path = tmp_path / "test.log"
        monkeypatch.setenv("AIACCEL_JOB_CONFIG", str(self.config_test_path))

        subprocess.run(
            self.cmd + ["cpu", log_path, "--", "sleep", "0"],
            check=True,
        )

        with open(tmp_path / "config_path.txt") as f:
            assert Path(f.read().rstrip("\n")) == self.config_test_path

    def test_cpu(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        log_path = tmp_path / "test.log"

        subprocess.run(
            self.make_command() + ["cpu", log_path, "--", "echo", "hello"],
            check=True,
        )

        assert log_path.exists()
        assert log_path.read_text().strip().endswith("hello")

    # TODO: Add tests for log filenames containing shell metacharacters
    # and other characters that require escaping.
    def test_cpu_log_filename_with_spaces(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        log_path = tmp_path / "test log.log"

        subprocess.run(
            self.make_command() + ["cpu", log_path, "--", "echo", "hello"],
            check=True,
        )

        assert log_path.exists()
        assert log_path.read_text().strip().endswith("hello")
