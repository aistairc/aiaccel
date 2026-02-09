from __future__ import annotations

from typing import Any

from collections.abc import Callable

import pytest

from aiaccel.hpo.modelbridge.config import BridgeConfig, load_bridge_config


def test_load_bridge_config_roundtrip(make_bridge_config: Callable[[str], dict[str, Any]]) -> None:
    data = make_bridge_config("./work/tmp")
    config = load_bridge_config(data)
    assert isinstance(config, BridgeConfig)
    assert config.bridge.output_dir.as_posix().endswith("work/tmp")
    assert config.bridge.strict_mode is False
    assert config.bridge.scenarios[0].metrics == ["mae", "mse", "r2"]


@pytest.mark.parametrize(
    "mutator",
    [
        lambda cfg: cfg["bridge"]["scenarios"][0]["train_params"].pop("macro"),
        lambda cfg: cfg["bridge"]["scenarios"][0].update({"metrics": "mae"}),
    ],
)
def test_load_bridge_config_validation_error(
    make_bridge_config: Callable[[str], dict[str, Any]], mutator: Callable[[dict[str, Any]], None]
) -> None:
    data = make_bridge_config("./work/tmp")
    mutator(data)
    with pytest.raises(ValueError):
        load_bridge_config(data)


def test_load_bridge_config_unknown_metrics(make_bridge_config: Callable[[str], dict[str, Any]]) -> None:
    data = make_bridge_config("./work/tmp")
    data["bridge"]["scenarios"][0]["metrics"] = ["mae", "unknown"]
    with pytest.raises(ValueError):
        load_bridge_config(data)


def test_load_bridge_config_regression_overrides(make_bridge_config: Callable[[str], dict[str, Any]]) -> None:
    data = make_bridge_config("./work/tmp")
    scenario = data["bridge"]["scenarios"][0]
    scenario["metrics"] = ["mae"]
    # Flat structure for regression config
    scenario["regression"] = {
        "kind": "gpr",
        "degree": 3,
        "kernel": "Matern52",
        "noise": 1.2e-4,
    }
    config = load_bridge_config(data)
    regression = config.bridge.scenarios[0].regression
    assert regression.kind == "gpr"
    assert regression.degree == 3
    assert regression.kernel == "Matern52"
    assert regression.noise == pytest.approx(1.2e-4)


def test_load_bridge_config_regression_aliases(make_bridge_config: Callable[[str], dict[str, Any]]) -> None:
    data = make_bridge_config("./work/tmp")
    scenario = data["bridge"]["scenarios"][0]
    # 'type' is aliased to 'kind' in config.py
    scenario["regression"] = {"type": "gpr", "noise": 9.9e-5}
    config = load_bridge_config(data)
    regression = config.bridge.scenarios[0].regression
    assert regression.kind == "gpr"
    assert regression.noise == pytest.approx(9.9e-5)
