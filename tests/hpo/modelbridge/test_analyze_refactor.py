from __future__ import annotations

from typing import Any

from collections.abc import Callable
from pathlib import Path

import pytest

import aiaccel.hpo.modelbridge.analyze as analyze_module
from aiaccel.hpo.modelbridge.common import write_json
from aiaccel.hpo.modelbridge.config import BridgeConfig, load_bridge_config


def _config(tmp_path: Path, make_bridge_config: Callable[[str], dict[str, Any]]) -> BridgeConfig:
    payload = make_bridge_config(str(tmp_path / "outputs"))
    payload["hpo"]["base_config"] = str(tmp_path / "optimize.yaml")
    (tmp_path / "optimize.yaml").write_text("optimize:\n  goal: minimize\n", encoding="utf-8")
    return load_bridge_config(payload)


def test_fit_regression_uses_internal_fit_helper(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, make_bridge_config)
    scenario = config.bridge.scenarios[0]
    scenario_path = config.bridge.output_dir / scenario.name
    scenario_path.mkdir(parents=True, exist_ok=True)
    (scenario_path / "train_pairs.csv").write_text("run_id,macro_x,micro_y\n0,0.0,1.0\n1,1.0,3.0\n", encoding="utf-8")

    observed: dict[str, Any] = {}

    def fake_fit(
        features: list[dict[str, float]],
        targets: list[dict[str, float]],
        _config: Any,
    ) -> dict[str, Any]:
        observed["features"] = features
        observed["targets"] = targets
        return {
            "kind": "linear",
            "feature_names": ["x"],
            "target_names": ["y"],
            "degree": 1,
            "coefficients": [[1.0]],
            "intercept": [0.0],
        }

    monkeypatch.setattr(analyze_module, "_fit_regression", fake_fit)
    result = analyze_module.fit_regression(config)

    assert result.status == "success"
    assert len(observed["features"]) == 2
    assert len(observed["targets"]) == 2


def test_evaluate_model_uses_internal_predict_helper(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, make_bridge_config)
    scenario = config.bridge.scenarios[0]
    scenario_path = config.bridge.output_dir / scenario.name
    scenario_path.mkdir(parents=True, exist_ok=True)
    write_json(
        scenario_path / "models" / "regression_model.json",
        {
            "kind": "linear",
            "feature_names": ["x"],
            "target_names": ["y"],
            "degree": 1,
            "coefficients": [[1.0]],
            "intercept": [0.0],
        },
    )
    (scenario_path / "test_pairs.csv").write_text("run_id,macro_x,micro_y\n0,0.5,2.0\n1,0.8,2.6\n", encoding="utf-8")

    calls = {"count": 0}

    def fake_predict(model_payload: dict[str, Any], features: list[dict[str, float]]) -> list[dict[str, float]]:
        _ = model_payload
        calls["count"] += 1
        return [{"y": item["x"]} for item in features]

    monkeypatch.setattr(analyze_module, "_predict_regression", fake_predict)
    result = analyze_module.evaluate_model(config)

    assert result.status == "success"
    assert calls["count"] == 1
