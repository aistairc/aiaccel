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

    process = subprocess.Popen(
        cmd
        + [
            "cpu",
            log_path,
            "--",
            "sleep",
            "30",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert process.stdout is not None

    line = process.stdout.readline().strip()
    print(f"aiaccel-job stdout: {line!r}")

    job_id = line.split()[-1]

    for _ in range(30):
        result = subprocess.run(
            ["squeue", "-j", job_id, "-h", "-o", "%T %R"],
            capture_output=True,
            text=True,
        )

        print(f"{result.stdout=}")

        if result.stdout.startswith("RUNNING"):
            break

        time.sleep(1)
    else:
        result = subprocess.run(
            ["scontrol", "show", "job", job_id],
            capture_output=True,
            text=True,
        )
        print(result.stdout)

        process.terminate()
        process.wait(timeout=5)

        pytest.fail(f"Slurm job {job_id} did not enter RUNNING state")
