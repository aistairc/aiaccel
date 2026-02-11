from __future__ import annotations

from typing import Any

from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from aiaccel.hpo.modelbridge import api
from aiaccel.hpo.modelbridge.config import BridgeConfig, load_bridge_config
from aiaccel.hpo.modelbridge.toolkit.io import read_json
from aiaccel.hpo.modelbridge.toolkit.results import PipelineResult, StepResult


def _write_config(tmp_path: Path, make_bridge_config: Callable[[str], dict[str, Any]]) -> Path:
    payload = make_bridge_config(str(tmp_path / "outputs"))
    payload["hpo"]["base_config"] = str(tmp_path / "optimize.yaml")
    (tmp_path / "optimize.yaml").write_text("optimize:\n  goal: minimize\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config_path


def test_api_load_config(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
) -> None:
    config_path = _write_config(tmp_path, make_bridge_config)
    config = api.load_config(config_path, overrides={"bridge": {"seed": 123}})
    assert isinstance(config, BridgeConfig)
    assert config.bridge.seed == 123


def test_api_run_invokes_pipeline(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = make_bridge_config(str(tmp_path / "outputs"))
    payload["hpo"]["base_config"] = str(tmp_path / "optimize.yaml")
    (tmp_path / "optimize.yaml").write_text("optimize:\n  goal: minimize\n", encoding="utf-8")
    config = load_bridge_config(payload)
    calls: dict[str, object] = {}

    def fake_setup_logging(*_args: object, **_kwargs: object) -> Path:
        calls["logged"] = True
        return tmp_path / "log.txt"

    def fake_run_pipeline(
        _config: BridgeConfig,
        steps: list[str] | None = None,
        *,
        profile: str | None = None,
        train_db_paths: list[Path] | None = None,
        eval_db_paths: list[Path] | None = None,
        train_db_pairs: list[tuple[Path, Path]] | None = None,
        eval_db_pairs: list[tuple[Path, Path]] | None = None,
    ) -> Any:
        calls["steps"] = steps
        calls["profile"] = profile
        calls["train_db_paths"] = train_db_paths
        _ = eval_db_paths
        _ = train_db_pairs
        _ = eval_db_pairs
        return PipelineResult(results=[])

    monkeypatch.setattr(api, "setup_logging", fake_setup_logging)
    monkeypatch.setattr(api, "run_pipeline", fake_run_pipeline)

    result = api.run(
        config,
        profile="prepare",
        train_db_paths=[tmp_path / "a.db"],
    )
    assert isinstance(result, PipelineResult)
    assert result.results == []
    assert calls["profile"] == "prepare"
    assert calls["train_db_paths"] == [tmp_path / "a.db"]


def test_api_step_callables_exist() -> None:
    expected_names = [
        "prepare_train_step",
        "prepare_eval_step",
        "collect_train_step",
        "collect_eval_step",
        "fit_regression_step",
        "evaluate_model_step",
        "publish_summary_step",
        "emit_commands_step",
    ]
    for name in expected_names:
        assert callable(getattr(api, name))


def test_api_prepare_train_step_executes(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
) -> None:
    config_path = _write_config(tmp_path, make_bridge_config)
    config = api.load_config(config_path)

    result = api.prepare_train_step(config, enable_logging=False)
    assert result.step == "prepare_train"
    assert result.status == "success"

    state = read_json(config.bridge.output_dir / "workspace" / "state" / "prepare_train.json")
    assert state["status"] == "success"


def test_api_emit_commands_step_accepts_execution_target(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path, make_bridge_config)
    config = api.load_config(config_path)
    calls: dict[str, object] = {}

    def fake_emit(
        _config: BridgeConfig,
        role: str,
        fmt: str,
        execution_target: str | None = None,
    ) -> Path:
        calls["role"] = role
        calls["fmt"] = fmt
        calls["target"] = execution_target
        return config.bridge.output_dir / "workspace" / "commands" / "train.sh"

    monkeypatch.setattr(api, "emit_commands", fake_emit)
    path = api.emit_commands_step(
        config,
        role="train",
        fmt="shell",
        execution_target="abci",
        enable_logging=False,
    )

    assert calls == {"role": "train", "fmt": "shell", "target": "abci"}
    assert path.name == "train.sh"


def test_api_step_wrappers_use_common_logging_helper(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path, make_bridge_config)
    config = api.load_config(config_path)
    calls: list[bool] = []

    def fake_run_with_logging(
        _config: BridgeConfig,
        *,
        enable_logging: bool,
        action: Callable[[], StepResult],
    ) -> StepResult:
        calls.append(enable_logging)
        return action()

    monkeypatch.setattr(api, "_run_with_optional_logging", fake_run_with_logging)
    monkeypatch.setattr(api, "prepare_train", lambda _config: StepResult(step="prepare_train", status="success"))
    monkeypatch.setattr(api, "fit_regression", lambda _config: StepResult(step="fit_regression", status="success"))

    prepare_result = api.prepare_train_step(config, enable_logging=False)
    fit_result = api.fit_regression_step(config, enable_logging=True)

    assert prepare_result.step == "prepare_train"
    assert fit_result.step == "fit_regression"
    assert calls == [False, True]
