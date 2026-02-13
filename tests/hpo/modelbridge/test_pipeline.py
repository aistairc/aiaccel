from __future__ import annotations

from typing import Any

from collections.abc import Callable
import inspect
from pathlib import Path

import pytest

from aiaccel.hpo.modelbridge.common import StepResult, read_json
from aiaccel.hpo.modelbridge.config import BridgeConfig, load_bridge_config
import aiaccel.hpo.modelbridge.pipeline as pipeline_module
from aiaccel.hpo.modelbridge.pipeline import run_pipeline, steps_for_profile
from aiaccel.hpo.modelbridge.prepare import prepare_train


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


def test_run_pipeline_rejects_unknown_profile(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
) -> None:
    config = _load_config(tmp_path, make_bridge_config)
    with pytest.raises(ValueError, match="Unknown profile"):
        run_pipeline(config, profile="invalid")


def test_pipeline_step_dispatch_uses_explicit_registry_map() -> None:
    assert tuple(pipeline_module.STEP_ACTIONS.keys()) == tuple(steps_for_profile("full"))
    assert "globals().get" not in inspect.getsource(pipeline_module)


def test_run_pipeline_full_requires_external_outputs(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
) -> None:
    config = _load_config(tmp_path, make_bridge_config)
    with pytest.raises(RuntimeError, match="Full profile requires external HPO outputs"):
        run_pipeline(config, profile="full")

    # prepare profile portion must have run before readiness validation fails.
    assert (config.bridge.output_dir / "workspace" / "state" / "prepare_train.json").exists()
    assert (config.bridge.output_dir / "workspace" / "state" / "prepare_eval.json").exists()
    assert not (config.bridge.output_dir / "workspace" / "state" / "collect_train.json").exists()


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
        "sampler_seed",
        "optimizer_seed",
        "seed_mode",
        "execution_target",
        "seed",
        "objective_command",
    }.issubset(entry.keys())
    assert Path(entry["config_path"]).exists()
    assert isinstance(entry["sampler_seed"], int)
    assert isinstance(entry["optimizer_seed"], int)


def test_prepare_profile_uses_user_defined_seed_values_end_to_end(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
) -> None:
    payload = _config_payload(tmp_path, make_bridge_config)
    payload["bridge"]["train_runs"] = 2
    payload["bridge"]["eval_runs"] = 1
    payload["bridge"]["seed_policy"] = {
        "sampler": {
            "mode": "user_defined",
            "user_values": {
                "train_macro": [101, 102],
                "train_micro": [201, 202],
                "eval_macro": [301],
                "eval_micro": [401],
            },
        },
        "optimizer": {
            "mode": "user_defined",
            "user_values": {
                "train_macro": [1001, 1002],
                "train_micro": [2001, 2002],
                "eval_macro": [3001],
                "eval_micro": [4001],
            },
        },
    }
    config = load_bridge_config(payload)

    run_pipeline(config, profile="prepare")
    train_entries = read_json(config.bridge.output_dir / "workspace" / "train_plan.json")["entries"]
    eval_entries = read_json(config.bridge.output_dir / "workspace" / "eval_plan.json")["entries"]

    expected_train = {
        (0, "macro"): (101, 1001),
        (0, "micro"): (201, 2001),
        (1, "macro"): (102, 1002),
        (1, "micro"): (202, 2002),
    }
    expected_eval = {
        (0, "macro"): (301, 3001),
        (0, "micro"): (401, 4001),
    }

    observed_train = {
        (item["run_id"], item["target"]): (item["sampler_seed"], item["optimizer_seed"]) for item in train_entries
    }
    observed_eval = {
        (item["run_id"], item["target"]): (item["sampler_seed"], item["optimizer_seed"]) for item in eval_entries
    }

    assert observed_train == expected_train
    assert observed_eval == expected_eval
    assert all(item["seed_mode"] == "user_defined" for item in train_entries)
    assert all(item["seed_mode"] == "user_defined" for item in eval_entries)


def test_prepare_emits_commands_when_configured(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
) -> None:
    payload = _config_payload(tmp_path, make_bridge_config)
    payload["bridge"]["execution"] = {"emit_on_prepare": True, "target": "local"}
    config = load_bridge_config(payload)

    result = prepare_train(config)

    assert result.status == "success"
    command_path = config.bridge.output_dir / "workspace" / "commands" / "train.sh"
    assert command_path.exists()
    assert "aiaccel-hpo optimize --config" in command_path.read_text(encoding="utf-8")


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

    monkeypatch.setattr("aiaccel.hpo.modelbridge.collect._pairs_from_paths", fake_scan)
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
