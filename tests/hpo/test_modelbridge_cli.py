from __future__ import annotations

from typing import Any

from collections.abc import Callable
from pathlib import Path
import shutil
from unittest.mock import patch

import pytest
import yaml

from aiaccel.hpo.apps.modelbridge import main as cli_main
from aiaccel.hpo.modelbridge import api
from aiaccel.hpo.modelbridge.toolkit.io import read_json


def _write_config(tmp_path: Path, make_bridge_config: Callable[[str], dict[str, Any]]) -> Path:
    payload = make_bridge_config(str(tmp_path / "outputs"))
    payload["hpo"]["base_config"] = str(tmp_path / "optimize.yaml")
    (tmp_path / "optimize.yaml").write_text("optimize:\n  goal: minimize\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config_path


def test_modelbridge_cli_validate(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    make_bridge_config: Callable[[str], dict[str, Any]],
) -> None:
    config_path = _write_config(tmp_path, make_bridge_config)
    cli_main(["validate", "--config", str(config_path)])
    assert "outputs" not in capsys.readouterr().out


def test_modelbridge_cli_output_dir_arg(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
) -> None:
    config_path = _write_config(tmp_path, make_bridge_config)
    override_output = tmp_path / "custom_output"

    with patch("aiaccel.hpo.apps.modelbridge.api.run") as mock_run:
        cli_main(["run", "--config", str(config_path), "--output_dir", str(override_output), "--profile", "prepare"])

        assert mock_run.called
        bridge_config = mock_run.call_args[0][0]
        assert bridge_config.bridge.output_dir == override_output


def test_modelbridge_cli_steps_profile_mutual_exclusive(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
) -> None:
    config_path = _write_config(tmp_path, make_bridge_config)
    with pytest.raises(SystemExit):
        cli_main(
            [
                "run",
                "--config",
                str(config_path),
                "--steps",
                "prepare_train",
                "--profile",
                "prepare",
            ]
        )


def test_modelbridge_cli_collect_train_db_paths(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
) -> None:
    config_path = _write_config(tmp_path, make_bridge_config)
    path1 = tmp_path / "a.db"
    path2 = tmp_path / "b.db"

    with patch("aiaccel.hpo.apps.modelbridge.api.run") as mock_run:
        cli_main(
            [
                "collect-train",
                "--config",
                str(config_path),
                "--train-db-path",
                str(path1),
                "--train-db-path",
                str(path2),
            ]
        )

        kwargs = mock_run.call_args.kwargs
        assert kwargs["steps"] == ["collect_train"]
        assert kwargs["train_db_paths"] == [path1.resolve(), path2.resolve()]


def test_modelbridge_cli_emit_commands(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_config(tmp_path, make_bridge_config)
    command_file = tmp_path / "outputs" / "workspace" / "commands" / "train.sh"

    with patch("aiaccel.hpo.apps.modelbridge.api.emit_commands_step", return_value=command_file):
        cli_main(
            [
                "emit-commands",
                "--config",
                str(config_path),
                "--role",
                "train",
                "--format",
                "shell",
            ]
        )

    assert str(command_file) in capsys.readouterr().out


def test_modelbridge_cli_emit_commands_with_execution_target(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_config(tmp_path, make_bridge_config)
    command_file = tmp_path / "outputs" / "workspace" / "commands" / "train.sh"

    with patch("aiaccel.hpo.apps.modelbridge.api.emit_commands_step", return_value=command_file) as mock_emit:
        cli_main(
            [
                "emit-commands",
                "--config",
                str(config_path),
                "--role",
                "train",
                "--format",
                "shell",
                "--execution-target",
                "abci",
            ]
        )

        assert mock_emit.call_args.kwargs["execution_target"] == "abci"
    assert str(command_file) in capsys.readouterr().out


def test_modelbridge_cli_run_prepare_emit_options(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
) -> None:
    config_path = _write_config(tmp_path, make_bridge_config)

    with patch("aiaccel.hpo.apps.modelbridge.api.run") as mock_run:
        cli_main(
            [
                "run",
                "--config",
                str(config_path),
                "--profile",
                "prepare",
                "--prepare-emit-commands",
                "--prepare-execution-target",
                "abci",
            ]
        )

        bridge_config = mock_run.call_args[0][0]
        assert bridge_config.bridge.execution.emit_on_prepare is True
        assert bridge_config.bridge.execution.target == "abci"


def test_modelbridge_cli_prepare_train_emit_options(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
) -> None:
    config_path = _write_config(tmp_path, make_bridge_config)

    with patch("aiaccel.hpo.apps.modelbridge.api.run") as mock_run:
        cli_main(
            [
                "prepare-train",
                "--config",
                str(config_path),
                "--emit-commands",
                "--execution-target",
                "abci",
            ]
        )

        bridge_config = mock_run.call_args[0][0]
        assert bridge_config.bridge.execution.emit_on_prepare is True
        assert bridge_config.bridge.execution.target == "abci"


def test_modelbridge_cli_prepare_emit_flags_require_prepare_profile(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
) -> None:
    config_path = _write_config(tmp_path, make_bridge_config)
    with pytest.raises(SystemExit):
        cli_main(
            [
                "run",
                "--config",
                str(config_path),
                "--profile",
                "analyze",
                "--prepare-emit-commands",
            ]
        )


def test_api_cli_equivalence_prepare(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
) -> None:
    config_path = _write_config(tmp_path, make_bridge_config)
    config = api.load_config(config_path)

    api.run(config, profile="prepare", enable_logging=False)
    api_plan = read_json(config.bridge.output_dir / "workspace" / "train_plan.json")

    shutil.rmtree(config.bridge.output_dir)
    cli_main(["run", "--config", str(config_path), "--profile", "prepare"])
    cli_plan = read_json(config.bridge.output_dir / "workspace" / "train_plan.json")

    assert api_plan["entries"] == cli_plan["entries"]
