from __future__ import annotations

from typing import Any

from collections.abc import Callable
from pathlib import Path

import pytest

import aiaccel.hpo.modelbridge.analyze as analyze_module
from aiaccel.hpo.modelbridge.config import BridgeConfig, RegressionConfig, load_bridge_config
from aiaccel.hpo.modelbridge.toolkit.io import write_json


def _config(tmp_path: Path, make_bridge_config: Callable[[str], dict[str, Any]]) -> BridgeConfig:
    payload = make_bridge_config(str(tmp_path / "outputs"))
    payload["hpo"]["base_config"] = str(tmp_path / "optimize.yaml")
    (tmp_path / "optimize.yaml").write_text("optimize:\n  goal: minimize\n", encoding="utf-8")
    return load_bridge_config(payload)


def test_fit_regression_uses_regressor_adapter(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, make_bridge_config)
    scenario = config.bridge.scenarios[0]
    scenario_path = config.bridge.output_dir / scenario.name
    scenario_path.mkdir(parents=True, exist_ok=True)
    (scenario_path / "train_pairs.csv").write_text("run_id,macro_x,micro_y\n0,0.0,1.0\n1,1.0,3.0\n", encoding="utf-8")
    observed_kinds: list[str] = []
    fit_calls = [0]

    class DummyAdapter:
        def fit(
            self,
            features: list[dict[str, float]],
            targets: list[dict[str, float]],
            _config: RegressionConfig,
        ) -> dict[str, Any]:
            fit_calls[0] += 1
            assert len(features) == len(targets) == 2
            return {
                "kind": "linear",
                "feature_names": ["x"],
                "target_names": ["y"],
                "degree": 1,
                "coefficients": [[1.0]],
                "intercept": [0.0],
            }

        def predict(self, model_payload: dict[str, Any], features: list[dict[str, float]]) -> list[dict[str, float]]:
            _ = model_payload
            return [{"y": item["x"]} for item in features]

    def fake_get_adapter(kind: str) -> DummyAdapter:
        observed_kinds.append(kind)
        return DummyAdapter()

    monkeypatch.setattr(analyze_module, "get_regressor_adapter", fake_get_adapter)
    result = analyze_module.fit_regression(config)

    assert result.status == "success"
    assert fit_calls[0] == 1
    assert observed_kinds == [scenario.regression.kind]


def test_evaluate_model_uses_regressor_adapter(
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
    observed_kinds: list[str] = []
    predict_calls = [0]

    class DummyAdapter:
        def fit(
            self,
            features: list[dict[str, float]],
            targets: list[dict[str, float]],
            _config: RegressionConfig,
        ) -> dict[str, Any]:
            _ = features
            _ = targets
            return {}

        def predict(self, model_payload: dict[str, Any], features: list[dict[str, float]]) -> list[dict[str, float]]:
            _ = model_payload
            predict_calls[0] += 1
            return [{"y": item["x"]} for item in features]

    def fake_get_adapter(kind: str) -> DummyAdapter:
        observed_kinds.append(kind)
        return DummyAdapter()

    monkeypatch.setattr(analyze_module, "get_regressor_adapter", fake_get_adapter)
    result = analyze_module.evaluate_model(config)

    assert result.status == "success"
    assert predict_calls[0] == 1
    assert observed_kinds == [scenario.regression.kind]
