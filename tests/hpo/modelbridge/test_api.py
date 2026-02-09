from __future__ import annotations

from typing import Any

from collections.abc import Callable
from pathlib import Path

import pytest
import yaml

from aiaccel.hpo.modelbridge import api
from aiaccel.hpo.modelbridge.config import BridgeConfig, load_bridge_config
from aiaccel.hpo.modelbridge.toolkit.io import read_json


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
        return {"ok": True}

    monkeypatch.setattr(api, "setup_logging", fake_setup_logging)
    monkeypatch.setattr(api, "run_pipeline", fake_run_pipeline)

    result = api.run(
        config,
        profile="prepare",
        train_db_paths=[tmp_path / "a.db"],
    )
    assert result == {"ok": True}
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
