from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aiaccel.hpo.modelbridge.config import load_bridge_config
from aiaccel.hpo.modelbridge.runner import execute_pipeline, plan_pipeline, run_pipeline


def _config(tmp_path: Path, make_bridge_config) -> dict[str, object]:
    data = make_bridge_config(tmp_path / "outputs")
    scenario = data["bridge"]["scenarios"][0]
    scenario["train_macro_trials"] = 3
    scenario["train_micro_trials"] = 3
    scenario["eval_macro_trials"] = 3
    scenario["eval_micro_trials"] = 3
    scenario["regression"] = {"degree": 1}
    data["bridge"]["seed"] = 7
    return data


def test_run_pipeline_dry_run(tmp_path: Path, make_bridge_config) -> None:
    bridge_config = load_bridge_config(_config(tmp_path, make_bridge_config))
    plan = plan_pipeline(bridge_config)
    payload = execute_pipeline(plan, dry_run=True)

    assert len(payload["contexts"]) == 7
    hpo_contexts = [ctx for ctx in payload["contexts"] if ctx["phase"] == "hpo"]
    assert any(ctx["role"] == "train" and ctx["target"] == "macro" for ctx in hpo_contexts)
    assert not (tmp_path / "outputs").exists()


def test_run_pipeline(tmp_path: Path, make_bridge_config) -> None:
    bridge_config = load_bridge_config(_config(tmp_path, make_bridge_config))
    summary = run_pipeline(bridge_config)

    summary_path = tmp_path / "outputs" / "summary.json"
    assert summary_path.exists()
    with summary_path.open("r", encoding="utf-8") as handle:
        summary_file: dict[str, Any] = json.load(handle)
    assert summary_file == summary

    scenario_summary = summary["demo"]
    assert "train_pairs" in scenario_summary
    scenario_dir = tmp_path / "outputs" / "scenarios" / "demo"
    regression_json = scenario_dir / "regression_train.json"
    metrics_json = scenario_dir / "regression_test_metrics.json"
    assert regression_json.exists()
    assert metrics_json.exists()
    metrics_payload = json.loads(metrics_json.read_text(encoding="utf-8"))
    assert "mae" in metrics_payload
    runs_root = tmp_path / "outputs" / "runs" / "demo"
    assert (runs_root / "train" / "macro" / "000" / "optuna.db").exists()
    assert (runs_root / "eval" / "micro" / "000" / "optuna.db").exists()


def test_run_pipeline_partial_phases(tmp_path: Path, make_bridge_config) -> None:
    bridge_config = load_bridge_config(_config(tmp_path, make_bridge_config))
    scenario_dir = tmp_path / "outputs" / "scenarios" / "demo"
    runs_root = tmp_path / "outputs" / "runs" / "demo"

    plan = plan_pipeline(bridge_config, phases=("hpo",))
    partial = execute_pipeline(plan)
    assert "contexts" in partial
    assert (runs_root / "train" / "macro" / "000" / "optuna.db").exists()
    assert not (scenario_dir / "regression_train.json").exists()

    execute_pipeline(plan_pipeline(bridge_config, phases=("regress",)))
    regression_path = scenario_dir / "regression_train.json"
    assert regression_path.exists()

    summary = execute_pipeline(plan_pipeline(bridge_config, phases=("summary",)))
    assert "demo" in summary and "train_pairs" in summary["demo"]


def test_run_pipeline_custom_metrics(tmp_path: Path, make_bridge_config) -> None:
    data = _config(tmp_path, make_bridge_config)
    data["bridge"]["scenarios"][0]["metrics"] = ["mae"]
    bridge_config = load_bridge_config(data)
    run_pipeline(bridge_config)
    metrics_path = tmp_path / "outputs" / "scenarios" / "demo" / "regression_test_metrics.json"
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert list(payload.keys()) == ["mae"]
