from __future__ import annotations

from typing import Any

from collections.abc import Callable
from pathlib import Path

import pytest

from aiaccel.hpo.modelbridge.config import BridgeConfig, load_bridge_config
from aiaccel.hpo.modelbridge.pipeline import run_pipeline
from aiaccel.hpo.modelbridge.prepare import prepare_train
from aiaccel.hpo.modelbridge.toolkit.io import read_json
from aiaccel.hpo.modelbridge.toolkit.results import StepResult


def _config_payload(tmp_path: Path, make_bridge_config: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
    payload = make_bridge_config(str(tmp_path / "outputs"))
    payload["hpo"]["base_config"] = str(tmp_path / "optimize.yaml")
    (tmp_path / "optimize.yaml").write_text("optimize:\n  goal: minimize\n", encoding="utf-8")
    return payload


def _load_config(tmp_path: Path, make_bridge_config: Callable[[str], dict[str, Any]]) -> BridgeConfig:
    return load_bridge_config(_config_payload(tmp_path, make_bridge_config))


def test_run_pipeline_default_profile_prepare(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
) -> None:
    config = _load_config(tmp_path, make_bridge_config)
    result = run_pipeline(config)

    assert [item.step for item in result.results] == ["prepare_train", "prepare_eval"]
    assert (config.bridge.output_dir / "workspace" / "train_plan.json").exists()
    assert (config.bridge.output_dir / "workspace" / "eval_plan.json").exists()
    assert (config.bridge.output_dir / "workspace" / "state" / "prepare_train.json").exists()
    assert (config.bridge.output_dir / "workspace" / "state" / "prepare_eval.json").exists()


def test_run_pipeline_steps_and_profile_are_exclusive(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
) -> None:
    config = _load_config(tmp_path, make_bridge_config)
    with pytest.raises(ValueError):
        run_pipeline(config, steps=["prepare_train"], profile="prepare")


def test_run_pipeline_does_not_call_subprocess(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _load_config(tmp_path, make_bridge_config)
    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fail_subprocess(*args: Any, **kwargs: Any) -> None:
        calls.append((args, kwargs))
        raise AssertionError("pipeline.py must not call subprocess.run")

    monkeypatch.setattr("subprocess.run", fail_subprocess)
    run_pipeline(config, profile="prepare")
    assert calls == []


def test_prepare_generates_configs_and_plan_entries(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
) -> None:
    config = _load_config(tmp_path, make_bridge_config)
    result = prepare_train(config)

    assert result.status == "success"
    plan = read_json(config.bridge.output_dir / "workspace" / "train_plan.json")
    entries = plan["entries"]
    assert entries
    entry = entries[0]
    assert {
        "scenario",
        "role",
        "run_id",
        "target",
        "config_path",
        "expected_db_path",
        "study_name",
        "seed",
        "objective_command",
    }.issubset(entry.keys())
    assert Path(entry["config_path"]).exists()


def test_collect_manifest_first(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _load_config(tmp_path, make_bridge_config)
    prepare_train(config)
    plan = read_json(config.bridge.output_dir / "workspace" / "train_plan.json")
    entries = plan["entries"]
    for entry in entries:
        db_path = Path(entry["expected_db_path"])
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_path.write_text("", encoding="utf-8")

    observed: dict[str, Any] = {}

    def fake_scan(
        _scenario: Any,
        _role: str,
        db_paths: list[Path],
    ) -> list[tuple[int, dict[str, float], dict[str, float]]]:
        observed["db_paths"] = [str(path) for path in db_paths]
        return [(0, {"x": 1.0}, {"y": 2.0})]

    monkeypatch.setattr("aiaccel.hpo.modelbridge.collect.scan_db_paths_for_pairs", fake_scan)
    result = run_pipeline(config, steps=["collect_train"]).results[0]
    assert result.status == "success"
    assert observed["db_paths"]


def test_collect_strict_mode_fails(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
) -> None:
    payload = _config_payload(tmp_path, make_bridge_config)
    payload["bridge"]["strict_mode"] = True
    config = load_bridge_config(payload)

    with pytest.raises(RuntimeError):
        run_pipeline(config, steps=["collect_train"])

    state = read_json(config.bridge.output_dir / "workspace" / "state" / "collect_train.json")
    assert state["status"] == "failed"


def test_fit_evaluate_publish_generate_artifacts(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
) -> None:
    config = _load_config(tmp_path, make_bridge_config)
    scenario_name = config.bridge.scenarios[0].name
    scenario_path = config.bridge.output_dir / scenario_name
    scenario_path.mkdir(parents=True, exist_ok=True)

    (scenario_path / "train_pairs.csv").write_text(
        "run_id,macro_x,micro_y\n0,0.0,1.0\n1,1.0,3.0\n",
        encoding="utf-8",
    )
    (scenario_path / "test_pairs.csv").write_text(
        "run_id,macro_x,micro_y\n0,0.5,2.0\n1,0.8,2.6\n",
        encoding="utf-8",
    )

    result = run_pipeline(
        config,
        steps=["fit_regression", "evaluate_model", "publish_summary"],
    )
    assert [item.status for item in result.results] == ["success", "success", "success"]
    assert (scenario_path / "models" / "regression_model.json").exists()
    assert (scenario_path / "metrics" / "train_metrics.json").exists()
    assert (scenario_path / "metrics" / "eval_metrics.json").exists()
    assert (scenario_path / "test_predictions.csv").exists()
    assert (config.bridge.output_dir / "summary.json").exists()
    assert (config.bridge.output_dir / "manifest.json").exists()
    for step in ["fit_regression", "evaluate_model", "publish_summary"]:
        assert (config.bridge.output_dir / "workspace" / "state" / f"{step}.json").exists()


def test_pipeline_delegates_analyze_steps(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _load_config(tmp_path, make_bridge_config)
    observed: list[str] = []

    def fake_fit(_config: BridgeConfig) -> StepResult:
        observed.append("fit")
        return StepResult(step="fit_regression", status="success")

    def fake_evaluate(_config: BridgeConfig) -> StepResult:
        observed.append("evaluate")
        return StepResult(step="evaluate_model", status="success")

    monkeypatch.setattr("aiaccel.hpo.modelbridge.pipeline.fit_regression", fake_fit)
    monkeypatch.setattr("aiaccel.hpo.modelbridge.pipeline.evaluate_model", fake_evaluate)

    result = run_pipeline(config, steps=["fit_regression", "evaluate_model"])
    assert [item.step for item in result.results] == ["fit_regression", "evaluate_model"]
    assert observed == ["fit", "evaluate"]
