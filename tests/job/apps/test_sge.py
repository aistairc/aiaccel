# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from pathlib import Path
import re
import subprocess
import time

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


def test_cpu_qdel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    log_path = tmp_path / "test.log"
    ready_path = tmp_path / "ready"
    config_path = Path(__file__).parent / "config" / "custom_sge.yaml"

    process = subprocess.Popen(
        cmd
        + [
            "--config",
            config_path,
            "cpu",
            log_path,
            "--",
            "bash",
            "-c",
            f"touch {ready_path}; sleep 10; exit 0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert process.stdout is not None

    # SGE qsub:
    # Your job 3 ("test") has been submitted
    line = process.stdout.readline().strip()

    match = re.search(r"Your job (\d+)", line)
    assert match is not None

    job_id = match.group(1)

    for _ in range(60):
        if ready_path.exists():
            break
        time.sleep(1)
    else:
        subprocess.run(["qdel", job_id], check=False)
        _, stderr = process.communicate(timeout=30)
        pytest.fail(f"SGE job {job_id} did not start: {stderr}")

    subprocess.run(["qdel", job_id], check=True)

    _, stderr = process.communicate(timeout=30)

    assert process.returncode == 1
    assert "Job failed with 143 exit code." in stderr
