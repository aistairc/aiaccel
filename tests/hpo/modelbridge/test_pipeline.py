from __future__ import annotations

from typing import Any

from collections.abc import Callable
from pathlib import Path
import subprocess

import pytest

from aiaccel.hpo.modelbridge.analyze import evaluate_model, fit_regression
from aiaccel.hpo.modelbridge.collect import collect_train
from aiaccel.hpo.modelbridge.common import read_json
from aiaccel.hpo.modelbridge.config import BridgeConfig, load_bridge_config
from aiaccel.hpo.modelbridge.hpo_runner import hpo_eval, hpo_train
from aiaccel.hpo.modelbridge.prepare import prepare_eval, prepare_train
from aiaccel.hpo.modelbridge.publish import publish_summary


def _config_payload(tmp_path: Path, make_bridge_config: Callable[[str], dict[str, Any]]) -> dict[str, Any]:
    payload = make_bridge_config(str(tmp_path / "outputs"))
    payload["hpo"]["base_config"] = str(tmp_path / "optimize.yaml")
    (tmp_path / "optimize.yaml").write_text("optimize:\n  goal: minimize\n", encoding="utf-8")
    return payload


def _load_config(tmp_path: Path, make_bridge_config: Callable[[str], dict[str, Any]]) -> BridgeConfig:
    return load_bridge_config(_config_payload(tmp_path, make_bridge_config))


def test_prepare_steps_generate_configs_and_plan_entries(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
) -> None:
    config = _load_config(tmp_path, make_bridge_config)

    train_result = prepare_train(config)
    eval_result = prepare_eval(config)

    assert train_result.status == "success"
    assert eval_result.status == "success"
    train_plan = read_json(config.bridge.output_dir / "workspace" / "train_plan.json")
    eval_plan = read_json(config.bridge.output_dir / "workspace" / "eval_plan.json")
    assert train_plan["entries"]
    assert eval_plan["entries"]
    assert (config.bridge.output_dir / "workspace" / "state" / "prepare_train.json").exists()
    assert (config.bridge.output_dir / "workspace" / "state" / "prepare_eval.json").exists()


def test_hpo_step_skips_when_plan_is_missing(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
) -> None:
    config = _load_config(tmp_path, make_bridge_config)
    result = hpo_train(config)
    assert result.status == "skipped"
    assert "Plan file not found" in (result.reason or "")
    state = read_json(config.bridge.output_dir / "workspace" / "state" / "hpo_train.json")
    assert state["status"] == "skipped"


def test_hpo_steps_generate_run_scripts_and_state(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _load_config(tmp_path, make_bridge_config)
    prepare_train(config)
    prepare_eval(config)

    def fake_run(_command: list[str], check: bool) -> subprocess.CompletedProcess[str]:
        assert check is True
        for role in ("train", "eval"):
            plan = config.bridge.output_dir / "workspace" / f"{role}_plan.json"
            if not plan.exists():
                continue
            for entry in read_json(plan)["entries"]:
                db = Path(entry["expected_db_path"])
                db.parent.mkdir(parents=True, exist_ok=True)
                db.write_text("", encoding="utf-8")
        return subprocess.CompletedProcess(args=_command, returncode=0)

    monkeypatch.setattr("aiaccel.hpo.modelbridge.hpo_runner.subprocess.run", fake_run)

    train_result = hpo_train(config)
    eval_result = hpo_eval(config)

    assert train_result.status == "success"
    assert eval_result.status == "success"
    train_shell = config.bridge.output_dir / "workspace" / "commands" / "train.sh"
    eval_shell = config.bridge.output_dir / "workspace" / "commands" / "eval.sh"
    assert train_shell.exists()
    assert eval_shell.exists()
    assert "bash " in train_shell.read_text(encoding="utf-8")

    plan = read_json(config.bridge.output_dir / "workspace" / "train_plan.json")
    first_cfg = Path(plan["entries"][0]["config_path"])
    assert (first_cfg.parent / "run.sh").exists()
    assert (first_cfg.parent / "run.json").exists()

    assert (config.bridge.output_dir / "workspace" / "state" / "hpo_train.json").exists()
    assert (config.bridge.output_dir / "workspace" / "state" / "hpo_eval.json").exists()


def test_hpo_step_failure_persists_failed_state(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _load_config(tmp_path, make_bridge_config)
    prepare_train(config)

    def fail_run(_command: list[str], check: bool) -> None:
        assert check is True
        raise subprocess.CalledProcessError(returncode=2, cmd="bash")

    monkeypatch.setattr("aiaccel.hpo.modelbridge.hpo_runner.subprocess.run", fail_run)

    with pytest.raises(RuntimeError, match="hpo_train execution failed"):
        hpo_train(config)

    state = read_json(config.bridge.output_dir / "workspace" / "state" / "hpo_train.json")
    assert state["status"] == "failed"


def test_prepare_imports_external_hpo_results(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
) -> None:
    payload = _config_payload(tmp_path, make_bridge_config)
    source_config = tmp_path / "external" / "config.yaml"
    source_db = tmp_path / "external" / "optuna.db"
    source_config.parent.mkdir(parents=True, exist_ok=True)
    source_config.write_text("optimize:\n  goal: minimize\n", encoding="utf-8")
    source_db.write_text("db", encoding="utf-8")
    payload["bridge"]["external_hpo_import"] = {
        "enabled": True,
        "entries": [
            {
                "scenario": payload["bridge"]["scenarios"][0]["name"],
                "role": "train",
                "run_id": 0,
                "target": "macro",
                "source_hpo_config": str(source_config),
                "source_optuna_db": str(source_db),
            }
        ],
    }
    config = load_bridge_config(payload)

    result = prepare_train(config)

    run_dir = config.bridge.output_dir / payload["bridge"]["scenarios"][0]["name"] / "runs" / "train" / "000" / "macro"
    assert result.outputs["num_imported_entries"] == 1
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "optuna.db").exists()


def test_collect_strict_mode_fails(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
) -> None:
    payload = _config_payload(tmp_path, make_bridge_config)
    payload["bridge"]["strict_mode"] = True
    config = load_bridge_config(payload)

    with pytest.raises(RuntimeError):
        collect_train(config)

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

    fit_result = fit_regression(config)
    eval_result = evaluate_model(config)
    publish_result = publish_summary(config)

    assert [fit_result.status, eval_result.status, publish_result.status] == ["success", "success", "success"]
    assert (scenario_path / "models" / "regression_model.json").exists()
    assert (scenario_path / "metrics" / "train_metrics.json").exists()
    assert (scenario_path / "metrics" / "eval_metrics.json").exists()
    assert (scenario_path / "test_predictions.csv").exists()
    assert (config.bridge.output_dir / "summary.json").exists()
    assert (config.bridge.output_dir / "manifest.json").exists()
