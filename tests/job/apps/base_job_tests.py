# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from abc import abstractmethod
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time

import pytest


class BaseJobTests:
    cmd: list[str]
    config_path: Path | None = None
    config_test_path: Path
    cancel_status: int
    run_cpu_array_nprocs: int = 2

    def make_command(self) -> list[str | Path]:
        command: list[str | Path] = list(self.cmd.copy())

        if self.config_path is not None:
            command += ["--config", self.config_path]

        return command

    def kill_job(self, pid: str) -> None:
        """Terminate the user program running inside the local job."""
        os.kill(int(pid), signal.SIGTERM)

    @abstractmethod
    def get_job_id(self, stdout: str) -> str:
        """Return the scheduler job ID from submit command output."""

    @abstractmethod
    def is_job_running(self, job_id: str) -> bool:
        """Return whether the scheduler job still exists."""

    @abstractmethod
    def cancel_job(self, job_id: str) -> None:
        """Cancel a scheduler job."""

    def wait_for_job_to_finish(self, job_id: str, timeout: int = 30) -> None:
        for _ in range(timeout):
            if not self.is_job_running(job_id):
                return

            time.sleep(1)

        pytest.fail(f"Job {job_id} did not terminate")

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

    # TODO: Add tests for log filenames containing shell metacharacters
    # and other characters that require escaping.
    @pytest.mark.parametrize(
        "log_filename",
        [
            "test.log",
            "test log.log",
        ],
    )
    def test_cpu(self, tmp_path: Path, log_filename: str, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        log_path = tmp_path / log_filename

        subprocess.run(
            self.make_command() + ["cpu", log_path, "--", "echo", "hello"],
            check=True,
        )

        assert log_path.exists()
        assert log_path.read_text().strip().endswith("hello")

    def test_cpu_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        log_path = tmp_path / "test.log"
        status_path = tmp_path / "test.out"

        process = subprocess.Popen(
            self.make_command() + ["cpu", log_path, "--", "bash", "-c", "echo before failure; exit 7"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        self.wait_run_cpu_failure(process)
        _, cpu_failure_stderr = process.communicate(timeout=60)

        assert process.returncode == 1

        assert log_path.exists()
        assert "before failure\n" in log_path.read_text()

        self.assert_run_cpu_failure(cpu_failure_stderr, status_path)

    def wait_run_cpu_failure(self, process: subprocess.Popen[str]) -> None:
        pass

    def assert_run_cpu_failure(self, cpu_failure_stderr: str, status_path: Path) -> None:
        pass

    def test_cpu_kill_job(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)

        log_path = tmp_path / "test.log"
        status_path = tmp_path / "test.out"
        ready_path = tmp_path / "ready"

        # Run job
        pid_path = ready_path.with_name(f"{ready_path.name}.pid")
        boot_id_path = ready_path.with_name(f"{ready_path.name}.boot_id")
        pid_namespace_path = ready_path.with_name(f"{ready_path.name}.pid_namespace")

        command = (
            f"echo $$ > {shlex.quote(str(pid_path))}"
            f" && cat /proc/sys/kernel/random/boot_id"
            f" > {shlex.quote(str(boot_id_path))}"
            f" && readlink /proc/$$/ns/pid"
            f" > {shlex.quote(str(pid_namespace_path))}"
            f" && touch {shlex.quote(str(ready_path))}"
            " && exec sleep 10"
        )

        process = subprocess.Popen(
            self.make_command() + ["cpu", log_path, "--", "bash", "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        for _ in range(60):
            if ready_path.exists():
                break

            if process.poll() is not None:
                _, stderr = process.communicate()
                pytest.fail(f"Job exited before starting: {stderr}")

            time.sleep(1)
        else:
            process.terminate()
            process.communicate(timeout=30)
            pytest.fail("User program did not start")

        try:
            # Check boot_id and PID
            job_boot_id = boot_id_path.read_text().strip()
            runner_boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            assert job_boot_id == runner_boot_id, (
                f"Job runs on a different kernel: job={job_boot_id}, pytest={runner_boot_id}"
            )

            job_pid_namespace = pid_namespace_path.read_text().strip()
            runner_pid_namespace = os.readlink("/proc/self/ns/pid")
            assert job_pid_namespace == runner_pid_namespace, (
                f"Job runs in a different PID namespace: job={job_pid_namespace}, pytest={runner_pid_namespace}"
            )
        except Exception:
            # Do not send a signal using an unverified PID.
            # The user program exits naturally after sleep 10.
            process.communicate(timeout=30)
            raise

        pid = pid_path.read_text().strip()
        self.kill_job(pid)

        process.communicate(timeout=30)

        assert process.returncode == 1

        assert log_path.exists()

        self.assert_cpu_kill_job(status_path)

    def assert_cpu_kill_job(self, status_path: Path) -> None:
        pass

    def test_cpu_cancel_job(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)

        log_path = tmp_path / "test.log"
        status_path = tmp_path / "test.out"
        ready_path = tmp_path / "ready"

        process = subprocess.Popen(
            self.make_command()
            + [
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

        job_id = self.get_job_id(process.stdout.readline())

        for _ in range(60):
            if ready_path.exists():
                break

            time.sleep(1)
        else:
            self.cancel_job(job_id)
            pytest.fail(f"Job {job_id} did not start")

        self.cancel_job(job_id)
        self.wait_for_job_to_finish(job_id)

        _, stderr = process.communicate(timeout=30)

        assert process.returncode == 1
        assert f"Job failed with {self.cancel_status} exit code." in stderr

        assert log_path.exists()

        assert status_path is not None
        assert status_path.exists()
        assert status_path.read_text().strip() == str(self.cancel_status)

    @pytest.mark.parametrize(
        "log_name",
        [
            pytest.param("test_log.log", id="default_filename"),
            pytest.param("test log.log", id="filename_with_spaces"),
        ],
    )
    def test_cpu_array(
        self,
        log_name: str,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        log_path = tmp_path / log_name

        self.run_cpu_array(log_path)

        self.assert_cpu_array(tmp_path, log_path)

    def run_cpu_array(self, log_path: Path) -> None:
        subprocess.run(
            self.make_command()
            + [
                "cpu",
                "--n_tasks",
                "2",
                "--n_tasks_per_proc",
                "1",
                "--n_procs",
                str(self.run_cpu_array_nprocs),
                log_path,
                "--",
                "bash",
                "-c",
                'echo "TASK_INDEX=$TASK_INDEX"',
            ],
            check=True,
        )

    def assert_cpu_array(self, tmp_path: Path, log_path: Path) -> None:
        log_prefix = log_path.with_suffix("")
        for process_index in [1, 2]:
            log_file = tmp_path / f"{log_prefix.name}.{process_index}.log"
            assert log_file.exists()
            assert log_file.read_text().strip() == f"TASK_INDEX={process_index}"

    def test_cpu_fails_when_log_write_fails(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)

        log_path = tmp_path / "test.log"
        log_path.symlink_to("/dev/full")
        status_path = tmp_path / "test.out"

        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            self.run_cpu_fails_when_log_write_fails(log_path)

        assert exc_info.value.returncode == 1
        self.assert_cpu_fails_when_log_write_fails(exc_info.value.stderr, status_path)

    def run_cpu_fails_when_log_write_fails(self, log_path: Path) -> None:
        subprocess.run(
            self.make_command() + ["cpu", log_path, "--", "bash", "-c", "printf 'job output\\n'"],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def assert_cpu_fails_when_log_write_fails(self, cpu_failure_stderr: str, status_path: Path) -> None:
        pass
