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


def test_cpu_sigterm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    log_path = tmp_path / "test.log"
    config_path = Path(__file__).parent / "config" / "custom_pbs.yaml"

    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(
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
            check=True,
        )


def test_cpu_qdel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    log_path = tmp_path / "test.log"
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
            "sleep 5; exit 0",
        ],
        stdout=subprocess.PIPE,
        text=True,
    )

    assert process.stdout is not None

    job_id = process.stdout.readline().strip()
    assert job_id

    for _ in range(30):
        result = subprocess.run(
            ["qstat", "-f", job_id],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0 and "job_state = R" in result.stdout:
            break

        time.sleep(1)
    else:
        pytest.fail(f"PBS job {job_id} did not enter the running state")

    subprocess.run(["qdel", job_id], check=True)

    returncode = process.wait(timeout=30)

    assert returncode != 0
