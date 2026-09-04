# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from pathlib import Path
import subprocess

import pytest

cmd = ["aiaccel-job", "sge"]


def test_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    log_path = tmp_path / "test.log"
    config_path = Path(__file__).parent / "config" / "custom_sge.yaml"

    subprocess.run(
        cmd + ["--config", config_path, "cpu", log_path, "--", "echo", "hello"],
        check=True,
    )

    assert log_path.read_text().strip().endswith("hello")


def test_cpu_sigterm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    log_path = tmp_path / "test.log"
    config_path = Path(__file__).parent / "config" / "custom_sge.yaml"

    result = subprocess.run(
        cmd
        + [
            "--config",
            config_path,
            "cpu",
            log_path,
            "--",
            "bash",
            "-c",
            "kill -TERM $PPID; sleep 1; exit 0",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Job failed with 1 exit code." in result.stderr
