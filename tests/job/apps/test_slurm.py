# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from pathlib import Path
import subprocess
import time

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


def test_popen(tmp_path: Path) -> None:
    script = tmp_path / "sleep.sh"
    script.write_text(
        """\
#!/bin/bash
#SBATCH -t 00:01:00

sleep 30
"""
    )

    result = subprocess.run(
        ["sbatch", script],
        capture_output=True,
        text=True,
        check=True,
    )

    job_id = result.stdout.split()[-1]

    time.sleep(2)

    result = subprocess.run(
        ["squeue", "-j", job_id, "-h", "-o", "%T %R"],
        capture_output=True,
        text=True,
        check=True,
    )

    print(result.stdout)


def test_cpu_scancel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    log_path = tmp_path / "test.log"

    ready_path = tmp_path / "ready"

    process = subprocess.Popen(
        cmd
        + [
            "cpu",
            log_path,
            "--",
            "bash",
            "-c",
            f"touch {ready_path}; trap '' TERM; sleep 30; exit 0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert process.stdout is not None

    job_id = process.stdout.readline().split()[-1]

    for _ in range(60):
        if ready_path.exists():
            break

        time.sleep(1)
    else:
        pytest.fail(f"Slurm job {job_id} did not start")

    subprocess.run(["scancel", job_id], check=True)

    _, stderr = process.communicate(timeout=30)

    assert process.returncode == 1
    assert "Job failed with 143 exit code." in stderr
