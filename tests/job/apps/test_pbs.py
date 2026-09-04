# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from pathlib import Path
import subprocess
import time

import pytest

cmd = ["aiaccel-job", "pbs"]


def test_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    log_path = tmp_path / "test.log"

    config_path = Path(__file__).parent / "config" / "custom_pbs.yaml"

    subprocess.run(
        cmd + ["--config", config_path, "cpu", log_path, "--", "echo", "hello"],
        check=True,
    )

    assert log_path.exists()
    assert log_path.read_text() == "hello\n"


def test_cpu_qdel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    log_path = tmp_path / "test.log"
    ready_path = tmp_path / "ready"
    config_path = Path(__file__).parent / "config" / "custom_pbs.yaml"

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
            f"touch {ready_path}; trap '' TERM; sleep 10; exit 0",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert process.stdout is not None

    # PBS qsub:
    # 1.hostname
    job_id = process.stdout.readline().strip()
    assert job_id

    for _ in range(60):
        if ready_path.exists():
            break
        time.sleep(1)
    else:
        subprocess.run(["qdel", job_id], check=False)
        _, stderr = process.communicate(timeout=30)
        pytest.fail(f"PBS job {job_id} did not start: {stderr}")

    subprocess.run(["qdel", job_id], check=True)

    _, stderr = process.communicate(timeout=30)

    assert process.returncode == 1
    assert "Job failed with 1 exit code." in stderr
