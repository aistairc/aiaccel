# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import os
from pathlib import Path
import subprocess

import pytest

cmd = ["aiaccel-job", "local"]


@pytest.mark.parametrize(
    "base_args",
    [
        ["cpu"],
        ["cpu", "--n_tasks=10"],
        ["gpu"],
        ["gpu", "--n_tasks=10"],
    ],
)
def test_default(base_args: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    log_path = tmp_path / "test.log"

    subprocess.run(cmd + base_args + [log_path, "--", "sleep", "0"], check=True)


def test_config_from_argparse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    log_path = tmp_path / "test.log"

    config_path = Path(__file__).parent / "config" / "custom_local.yaml"

    subprocess.run(cmd + ["--config", config_path, "cpu", log_path, "sleep", "0"], check=True)

    with open(tmp_path / "config_path.txt") as f:
        assert Path(f.read().rstrip("\n")) == Path(config_path)


def test_config_from_environ(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    log_path = tmp_path / "test.log"

    config_path = Path(__file__).parent / "config" / "custom_local.yaml"
    monkeypatch.setenv("AIACCEL_JOB_CONFIG", str(config_path))

    subprocess.run(cmd + ["cpu", log_path, "--", "sleep", "0"], check=True)

    with open(tmp_path / "config_path.txt") as f:
        assert Path(f.read().rstrip("\n")) == config_path


def test_local_log_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    log_path = tmp_path / "test.log"

    subprocess.run(cmd + ["cpu", log_path, "--", "echo", "test run"], check=True)

    log_output = log_path.read_text()
    assert "test run\n" in log_output


def test_local_fails_when_tee_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_tee = fake_bin / "tee"
    # Make tee fail deterministically after consuming the job output.
    fake_tee.write_text("#!/bin/sh\ncat >/dev/null\nexit 1\n")
    fake_tee.chmod(0o755)

    # Keep aiaccel-job and other system commands available.
    current_path = os.environ.get("PATH", "")
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{current_path}")

    log_path = tmp_path / "test.log"

    # The job succeeds; only tee should make the invocation fail.
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(
            cmd + ["cpu", log_path, "--", "echo", "job succeeded"],
            check=True,
        )
