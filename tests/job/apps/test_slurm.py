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


def test_cpu_scancel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    log_path = tmp_path / "test.log"

    process = subprocess.Popen(
        cmd
        + [
            "cpu",
            log_path,
            "--",
            "bash",
            "-c",
            "trap '' TERM; sleep 10; exit 0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert process.stdout is not None

    line = process.stdout.readline().strip()

    # sbatch output:
    # Submitted batch job 123
    fields = line.split()
    assert len(fields) >= 4
    job_id = fields[-1]

    for _ in range(30):
        result = subprocess.run(
            ["squeue", "-j", job_id, "-h", "-o", "%T"],
            capture_output=True,
            text=True,
            check=True,
        )

        if result.stdout.strip() == "RUNNING":
            break

        print(result.stdout.strip())

        time.sleep(1)
    else:
        pytest.fail(f"Slurm job {job_id} did not enter RUNNING state")

    subprocess.run(
        ["scancel", job_id],
        check=True,
    )

    stdout, stderr = process.communicate(timeout=30)

    assert process.returncode == 1
    assert "Job failed with 143 exit code." in stderr
