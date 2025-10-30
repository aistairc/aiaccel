from __future__ import annotations

from typing import Any

import pytest

from aiaccel.hpo.modelbridge.config import BridgeConfig, load_bridge_config
from aiaccel.hpo.modelbridge.exceptions import ValidationError


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
    with pytest.raises(ValidationError):
        load_bridge_config(data)


def test_load_bridge_config_invalid_metrics() -> None:
    data = _base_config()
    data["bridge"]["scenarios"][0]["metrics"] = "mae"
    with pytest.raises(ValidationError):
        load_bridge_config(data)
