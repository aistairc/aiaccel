from __future__ import annotations

from typing import Any

from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from aiaccel.hpo.modelbridge import api
from aiaccel.hpo.modelbridge.common import StepResult
from aiaccel.hpo.modelbridge.config import BridgeConfig


def _write_config(tmp_path: Path, make_bridge_config: Callable[[str], dict[str, Any]]) -> Path:
    payload = make_bridge_config(str(tmp_path / "outputs"))
    payload["hpo"]["base_config"] = str(tmp_path / "optimize.yaml")
    (tmp_path / "optimize.yaml").write_text("optimize:\n  goal: minimize\n", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return config_path


def _load_config(tmp_path: Path, make_bridge_config: Callable[[str], dict[str, Any]]) -> BridgeConfig:
    return api.load_config(_write_config(tmp_path, make_bridge_config))


def test_api_load_config(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
) -> None:
    config_path = _write_config(tmp_path, make_bridge_config)
    config = api.load_config(config_path, overrides={"bridge": {"seed": 123}})
    assert isinstance(config, BridgeConfig)
    assert config.bridge.seed == 123


def test_api_required_callables_exist() -> None:
    for name in [
        "load_config",
        "prepare_train_step",
        "prepare_eval_step",
        "hpo_train_step",
        "hpo_eval_step",
        "collect_train_step",
        "collect_eval_step",
        "fit_regression_step",
        "evaluate_model_step",
        "publish_summary_step",
    ]:
        assert callable(getattr(api, name))


def test_api_removed_pipeline_entrypoints_are_not_exposed() -> None:
    for name in ["run", "run_pipeline", "emit_commands_step"]:
        assert not hasattr(api, name)


def test_api_prepare_step_invokes_runtime_and_logging(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _load_config(tmp_path, make_bridge_config)
    calls: list[bool] = []

    def fake_setup_logging(*_args: object, **_kwargs: object) -> Path:
        calls.append(True)
        return tmp_path / "log.txt"

    monkeypatch.setattr(api, "setup_logging", fake_setup_logging)
    monkeypatch.setattr(
        api,
        "prepare_train",
        lambda *_args, **_kwargs: StepResult(step="prepare_train", status="success"),
    )
    result = api.prepare_train_step(config, enable_logging=True)
    assert result.step == "prepare_train"
    assert calls == [True]


def test_api_collect_step_forwards_db_paths(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _load_config(tmp_path, make_bridge_config)
    observed: dict[str, object] = {}

    def fake_collect(
        _config: BridgeConfig,
        db_paths: list[Path] | None = None,
        db_pairs: list[tuple[Path, Path]] | None = None,
    ) -> StepResult:
        observed["paths"] = db_paths
        observed["pairs"] = db_pairs
        return StepResult(step="collect_train", status="success")

    monkeypatch.setattr(api, "collect_train", fake_collect)
    path = tmp_path / "a.db"
    result = api.collect_train_step(config, db_paths=[path], enable_logging=False)
    assert result.step == "collect_train"
    assert observed["paths"] == [path]
    assert observed["pairs"] is None


def test_api_hpo_eval_step_invokes_runner(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _load_config(tmp_path, make_bridge_config)
    observed: dict[str, object] = {}

    def fake_hpo_eval(_config: BridgeConfig) -> StepResult:
        observed["called"] = True
        return StepResult(step="hpo_eval", status="success")

    monkeypatch.setattr(api, "hpo_eval", fake_hpo_eval)
    result = api.hpo_eval_step(config, enable_logging=False)
    assert result.step == "hpo_eval"
    assert observed["called"] is True
