from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from aiaccel.hpo.apps.modelbridge import main as cli_main


def _write_config(tmp_path: Path, make_bridge_config) -> Path:
    payload = make_bridge_config(tmp_path / "outputs")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config_path


def test_modelbridge_cli_plan(tmp_path: Path, capsys: pytest.CaptureFixture[str], make_bridge_config) -> None:
    config_path = _write_config(tmp_path, make_bridge_config)
    cli_main(["plan", "--config", str(config_path)])
    output = capsys.readouterr().out
    assert "contexts" in output
    assert not (tmp_path / "outputs").exists()


def test_modelbridge_cli_macro_phase(tmp_path: Path, make_bridge_config) -> None:
    config_path = _write_config(tmp_path, make_bridge_config)
    cli_main(["run", "--config", str(config_path), "--phase", "hpo", "--role", "train", "--target", "macro"])
    db_path = tmp_path / "outputs" / "runs" / "demo" / "train" / "macro" / "000" / "optuna.db"
    assert db_path.exists()


def test_modelbridge_cli_hpo_filters_without_phase(tmp_path: Path, make_bridge_config) -> None:
    config_path = _write_config(tmp_path, make_bridge_config)
    cli_main(["run", "--config", str(config_path), "--role", "train", "--target", "macro", "--run-id", "0"])
    db_path = tmp_path / "outputs" / "runs" / "demo" / "train" / "macro" / "000" / "optuna.db"
    assert db_path.exists()


def test_modelbridge_cli_validate(tmp_path: Path, capsys: pytest.CaptureFixture[str], make_bridge_config) -> None:
    config_path = _write_config(tmp_path, make_bridge_config)
    cli_main(["validate", "--config", str(config_path)])
    assert "outputs" not in capsys.readouterr().out
