from __future__ import annotations

import copy
from pathlib import Path

import pytest

from aiaccel.hpo.modelbridge.config import BridgeConfig, load_bridge_config


def test_load_bridge_config_roundtrip(make_bridge_config) -> None:
    data = make_bridge_config("./work/tmp")
    config = load_bridge_config(data)
    assert isinstance(config, BridgeConfig)
    assert config.bridge.output_dir.as_posix().endswith("work/tmp")
    assert config.bridge.scenarios[0].metrics == ("mae", "mse", "r2")


@pytest.mark.parametrize(
    "mutator",
    [
        lambda cfg: cfg["bridge"]["scenarios"][0]["train_params"].pop("macro"),
        lambda cfg: cfg["bridge"]["scenarios"][0].update({"metrics": "mae"}),
    ],
)
def test_load_bridge_config_validation_error(make_bridge_config, mutator) -> None:
    data = make_bridge_config("./work/tmp")
    mutator(data)
    with pytest.raises(ValueError):
        load_bridge_config(data)


def test_load_bridge_config_unknown_metrics(make_bridge_config) -> None:
    data = make_bridge_config("./work/tmp")
    data["bridge"]["scenarios"][0]["metrics"] = ["mae", "unknown"]
    with pytest.raises(ValueError):
        load_bridge_config(data)


def test_load_bridge_config_regression_overrides(make_bridge_config) -> None:
    data = make_bridge_config("./work/tmp")
    scenario = data["bridge"]["scenarios"][0]
    scenario["metrics"] = ["mae"]
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


def test_load_bridge_config_regression_aliases(make_bridge_config) -> None:
    data = make_bridge_config("./work/tmp")
    scenario = data["bridge"]["scenarios"][0]
    scenario["regression"] = {"type": "gpr", "alpha": 9.9e-5}
    config = load_bridge_config(data)
    regression = config.bridge.scenarios[0].regression
    assert regression.kind == "gpr"
    assert regression.noise == pytest.approx(9.9e-5)


def test_load_bridge_config_objective_timeout(make_bridge_config) -> None:
    data = make_bridge_config("./work/tmp")
    scenario = data["bridge"]["scenarios"][0]
    scenario["train_objective"]["timeout"] = 5.0
    config = load_bridge_config(data)
    scenario_cfg = config.bridge.scenarios[0]
    assert scenario_cfg.train_objective.timeout == pytest.approx(5.0)


def test_working_directory_defaults_to_output_dir(tmp_path: Path, make_bridge_config) -> None:
    data = make_bridge_config(tmp_path / "outputs")
    config = load_bridge_config(data)
    assert config.bridge.working_directory == config.bridge.output_dir


def test_working_directory_override(tmp_path: Path, make_bridge_config) -> None:
    workdir = tmp_path / "custom_workdir"
    data = make_bridge_config(tmp_path / "outputs")
    data["bridge"]["working_directory"] = str(workdir)
    config = load_bridge_config(data)
    assert config.bridge.working_directory == workdir
