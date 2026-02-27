from __future__ import annotations

from typing import Any, cast

from pathlib import Path

import pytest
import yaml

from aiaccel.hpo.modelbridge import prepare


def _write_config(config_path: Path) -> None:
    payload = {
        "n_train": 2,
        "n_test": 1,
        "objective_command": ["python", "objectives/simple_objective.py"],
        "train_macro_trials": 3,
        "train_micro_trials": 4,
        "test_macro_trials": 2,
        "test_micro_trials": 2,
        "train_params": {
            "macro": {"lr": {"low": 0.001, "high": 0.1, "log": True}},
            "micro": {
                "lr": {"low": 0.001, "high": 0.1, "log": True},
                "momentum": {"low": 0.1, "high": 0.9},
            },
        },
        "test_params": {
            "macro": {"lr": {"low": 0.001, "high": 0.1, "log": True}},
            "micro": {
                "lr": {"low": 0.001, "high": 0.1, "log": True},
                "momentum": {"low": 0.1, "high": 0.9},
            },
        },
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _load_generated_config(workspace: Path, role: str, target: str, run_id: str = "000") -> dict[str, Any]:
    config_path = workspace / "runs" / role / target / run_id / "config.yaml"
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return cast(dict[str, Any], loaded)


def test_run_prepare_generates_configs(tmp_path: Path) -> None:
    config_path = tmp_path / "config" / "config.yaml"
    objective_path = tmp_path / "objectives" / "simple_objective.py"
    objective_path.parent.mkdir(parents=True, exist_ok=True)
    objective_path.write_text("print('ok')\n", encoding="utf-8")
    _write_config(config_path)

    workspace = tmp_path / "workspace"
    n_train, n_test = prepare.run_prepare(config_path=config_path, workspace=workspace)

    assert (n_train, n_test) == (2, 1)
    train_macro = workspace / "runs" / "train" / "macro" / "000" / "config.yaml"
    test_micro = workspace / "runs" / "test" / "micro" / "000" / "config.yaml"
    assert train_macro.exists()
    assert test_micro.exists()

    payload = yaml.safe_load(train_macro.read_text(encoding="utf-8"))
    assert payload["n_trials"] == 3
    assert payload["study"]["study_name"] == "train-macro-000"
    assert payload["command"][1] == str(objective_path.resolve())
    assert "--lr={lr}" in payload["command"]
    assert payload["command"][-1] == "{out_filename}"
    assert payload["study"]["sampler"]["seed"] == 42

    test_macro_payload = _load_generated_config(workspace, "test", "macro")
    assert test_macro_payload["study"]["sampler"]["seed"] == 1042


def test_run_prepare_applies_seed_defaults_from_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config" / "config.yaml"
    objective_path = tmp_path / "objectives" / "simple_objective.py"
    objective_path.parent.mkdir(parents=True, exist_ok=True)
    objective_path.write_text("print('ok')\n", encoding="utf-8")
    _write_config(config_path)

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    payload["seed_defaults"] = {"train": 700, "test_macro": 1700, "test_micro": 1800}
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    workspace = tmp_path / "workspace"
    prepare.run_prepare(config_path=config_path, workspace=workspace)

    train_macro = _load_generated_config(workspace, "train", "macro")
    train_micro = _load_generated_config(workspace, "train", "micro")
    test_macro = _load_generated_config(workspace, "test", "macro")
    test_micro = _load_generated_config(workspace, "test", "micro")

    assert train_macro["study"]["sampler"]["seed"] == 700
    assert train_micro["study"]["sampler"]["seed"] == 800
    assert test_macro["study"]["sampler"]["seed"] == 1700
    assert test_micro["study"]["sampler"]["seed"] == 1800


def test_run_prepare_rejects_negative_runs(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("n_train: -1\nn_test: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="n_train"):
        prepare.run_prepare(config_path=config_path, workspace=tmp_path / "workspace")


def test_run_prepare_rejects_unknown_seed_defaults_key(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "n_train": 1,
                "n_test": 1,
                "objective_command": ["python", "objective.py"],
                "seed_defaults": {"unknown_key": 1},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unsupported seed_defaults key"):
        prepare.run_prepare(config_path=config_path, workspace=tmp_path / "workspace")
