# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from pathlib import Path
import subprocess

import pytest

cmd = ["aiaccel-job", "slurm"]


def test_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    log_path = tmp_path / "test.log"

    subprocess.run(
        cmd + ["cpu", log_path, "--", "echo", "hello"],
        check=True,
    )

    assert log_path.exists()
    assert "hello" in log_path.read_text()


def test_cpu_scancel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    log_path = tmp_path / "test.log"

    subprocess.run(
        cmd + ["cpu", log_path, "--", "trap '' TERM; sleep 10; exit 0"],
        text=True,
        check=True,
    )
