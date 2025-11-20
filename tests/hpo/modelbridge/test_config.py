from __future__ import annotations

from typing import Any

import pytest

from aiaccel.hpo.modelbridge.config import BridgeConfig, load_bridge_config


def _base_config() -> dict[str, Any]:
    return {
        "hpo": {"optimizer": "optuna", "sampler": "tpe"},
        "bridge": {
            "output_dir": "./work/tmp",
            "seed": 123,
            "scenarios": [
                {
                    "name": "demo",
                    "macro_trials": 2,
                    "micro_trials": 2,
                    "objective": {
                        "target": "tests.hpo.modelbridge.sample_objective.objective",
                    },
                    "params": {
                        "macro": {"x": {"low": 0, "high": 1}},
                        "micro": {"y": {"low": 0, "high": 1}},
                    },
                    "metrics": ["mae"],
                }
            ],
        },
    }


def test_load_bridge_config_roundtrip() -> None:
    config = load_bridge_config(_base_config())
    assert isinstance(config, BridgeConfig)
    assert config.bridge.output_dir.as_posix().endswith("work/tmp")
    assert config.bridge.scenarios[0].metrics == ("mae",)


def test_load_bridge_config_validation_error() -> None:
    data = _base_config()
    data["bridge"]["scenarios"][0]["params"].pop("macro")
    with pytest.raises(ValueError):
        load_bridge_config(data)


def test_load_bridge_config_invalid_metrics() -> None:
    data = _base_config()
    data["bridge"]["scenarios"][0]["metrics"] = "mae"
    with pytest.raises(ValueError):
        load_bridge_config(data)


def test_load_bridge_config_unknown_metrics() -> None:
    data = _base_config()
    data["bridge"]["scenarios"][0]["metrics"] = ["mae", "unknown"]
    with pytest.raises(ValueError):
        load_bridge_config(data)


def test_load_bridge_config_regression_overrides() -> None:
    data = _base_config()
    scenario = data["bridge"]["scenarios"][0]
    scenario["regression"] = {
        "kind": "gpr",
        "poly": {"degree": 3},
        "gpr": {"kernel": "Matern52", "noise": 1.2e-4},
    }
    config = load_bridge_config(data)
    regression = config.bridge.scenarios[0].regression
    assert regression.kind == "gpr"
    assert regression.degree == 3
    assert regression.kernel == "Matern52"
    assert regression.noise == pytest.approx(1.2e-4)


def test_load_bridge_config_regression_aliases() -> None:
    data = _base_config()
    scenario = data["bridge"]["scenarios"][0]
    scenario["regression"] = {"type": "gpr", "alpha": 9.9e-5}
    config = load_bridge_config(data)
    regression = config.bridge.scenarios[0].regression
    assert regression.kind == "gpr"
    assert regression.noise == pytest.approx(9.9e-5)


def test_load_bridge_config_objective_inheritance() -> None:
    data = _base_config()
    scenario = data["bridge"]["scenarios"][0]
    scenario["train_objective"] = {"timeout": 5.0}
    scenario["eval_objective"] = {"target": "tests.hpo.modelbridge.sample_objective.stateless_objective"}
    config = load_bridge_config(data)
    scenario_cfg = config.bridge.scenarios[0]
    assert scenario_cfg.train_objective is not None
    assert scenario_cfg.train_objective.target == scenario_cfg.objective.target
    assert scenario_cfg.train_objective.timeout == pytest.approx(5.0)
    assert scenario_cfg.eval_objective is not None
    assert scenario_cfg.eval_objective.target.endswith("stateless_objective")
    assert scenario_cfg.eval_objective.command is None


def test_load_bridge_config_params_inheritance() -> None:
    data = _base_config()
    scenario = data["bridge"]["scenarios"][0]
    scenario["train_params"] = {"macro": {"x": {"low": 0.5, "high": 1.5}}}
    config = load_bridge_config(data)
    scenario_cfg = config.bridge.scenarios[0]
    assert scenario_cfg.train_params is not None
    assert scenario_cfg.train_params.macro["x"].low == pytest.approx(0.5)
    assert "y" in scenario_cfg.train_params.micro
    assert scenario_cfg.train_params.micro["y"].low == pytest.approx(0.0)


def test_storage_default_none() -> None:
    config = load_bridge_config(_base_config())
    assert config.bridge.storage is None
