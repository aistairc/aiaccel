from __future__ import annotations

from pathlib import Path

import yaml
import pytest
from aiaccel.hpo.apps.modelbridge import main as cli_main


def _write_config(tmp_path: Path) -> Path:
    payload = {
        "bridge": {
            "output_dir": str(tmp_path / "outputs"),
            "seed": 5,
            "scenarios": [
                {
                    "name": "demo",
                    "macro_trials": 2,
                    "micro_trials": 2,
                    "objective": {
                        "target": "tests.hpo.modelbridge.sample_objective.objective",
                    },
                    "params": {
                        "macro": {"x": {"low": 0.0, "high": 1.0}},
                        "micro": {"y": {"low": 0.0, "high": 1.0}},
                    },
                }
            ],
        }
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config_path


def test_modelbridge_cli_dry_run(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    cli_main(["--config", str(config_path), "--dry-run"])


def test_modelbridge_cli_macro_phase(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    cli_main(["--config", str(config_path), "--phase", "hpo", "--role", "train", "--target", "macro"])
    db_path = tmp_path / "outputs" / "runs" / "demo" / "train" / "macro" / "000" / "optuna.db"
    assert db_path.exists()


def test_modelbridge_cli_hpo_filters_without_phase(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    cli_main(["--config", str(config_path), "--role", "train", "--target", "macro", "--run-id", "0"])
    db_path = tmp_path / "outputs" / "runs" / "demo" / "train" / "macro" / "000" / "optuna.db"
    assert db_path.exists()


def test_modelbridge_cli_rejects_old_phase(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    with pytest.raises(SystemExit):
        cli_main(["--config", str(config_path), "--phase", "train-macro"])
