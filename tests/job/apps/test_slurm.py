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
    status_path = tmp_path / "test.out"
    ready_path = tmp_path / "ready"

    process = subprocess.Popen(
        cmd
        + [
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

    assert status_path.exists()
    assert status_path.read_text().strip() == "143"


def test_cpu_log_filename_with_spaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    log_path = tmp_path / "test log.log"

    subprocess.run(
        cmd
        + [
            "cpu",
            log_path,
            "--",
            "echo",
            "hello",
        ],
        check=True,
    )

    assert log_path.exists()
    assert log_path.read_text().strip().endswith("hello")


def test_cpu_array(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    log_path = tmp_path / "test_log.log"

    subprocess.run(
        cmd
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

    for task_index in [1, 2]:
        log_file = tmp_path / f"test_log.{task_index}-1.log"

        assert log_file.exists()
        assert log_file.read_text().strip() == f"TASK_INDEX={task_index}"


def test_cpu_array_log_filename_with_spaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    log_path = tmp_path / "test log.log"

    subprocess.run(
        cmd
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

    for task_index in [1, 2]:
        log_file = tmp_path / f"test log.{task_index}-1.log"

        assert log_file.exists()
        assert log_file.read_text().strip() == f"TASK_INDEX={task_index}"
