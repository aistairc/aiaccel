from __future__ import annotations

from typing import Any

from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from aiaccel.hpo.apps.modelbridge import main as cli_main


def _write_config(tmp_path: Path, make_bridge_config: Callable[[str], dict[str, Any]]) -> Path:
    payload = make_bridge_config(str(tmp_path / "outputs"))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config_path


def test_modelbridge_cli_validate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], make_bridge_config: Callable[[str], dict[str, Any]]
) -> None:
    config_path = _write_config(tmp_path, make_bridge_config)
    cli_main(["validate", "--config", str(config_path)])
    assert "outputs" not in capsys.readouterr().out


def test_modelbridge_cli_overrides_set(tmp_path: Path, make_bridge_config: Callable[[str], dict[str, Any]]) -> None:
    config_path = _write_config(tmp_path, make_bridge_config)
    override_output = tmp_path / "custom_outputs"

    with patch("aiaccel.hpo.apps.modelbridge.run_pipeline") as mock_run:
        cli_main(
            [
                "run",
                "--config",
                str(config_path),
                "--set",
                f"bridge.output_dir={override_output}",
            ]
        )

        assert mock_run.called
        call_args = mock_run.call_args
        bridge_config = call_args[0][0]
        assert bridge_config.bridge.output_dir == override_output


def test_modelbridge_cli_output_dir_arg(tmp_path: Path, make_bridge_config: Callable[[str], dict[str, Any]]) -> None:
    config_path = _write_config(tmp_path, make_bridge_config)
    override_output = tmp_path / "arg_output_dir"

    with patch("aiaccel.hpo.apps.modelbridge.run_pipeline") as mock_run:
        cli_main(
            [
                "run",
                "--config",
                str(config_path),
                "--output_dir",
                str(override_output),
            ]
        )

        assert mock_run.called
        call_args = mock_run.call_args
        bridge_config = call_args[0][0]
        assert bridge_config.bridge.output_dir == override_output


def test_modelbridge_cli_steps(tmp_path: Path, make_bridge_config: Callable[[str], dict[str, Any]]) -> None:
    config_path = _write_config(tmp_path, make_bridge_config)

    with patch("aiaccel.hpo.apps.modelbridge.run_pipeline") as mock_run:
        cli_main(
            [
                "run",
                "--config",
                str(config_path),
                "--steps",
                "setup_train, regression",
            ]
        )

        assert mock_run.called
        call_args = mock_run.call_args
        steps = call_args[1].get("steps") or call_args[0][1]
        assert steps == ["setup_train", "regression"]


def test_modelbridge_cli_schema(capsys: pytest.CaptureFixture[str]) -> None:
    cli_main(["schema"])
    output = capsys.readouterr().out
    assert "properties" in output
