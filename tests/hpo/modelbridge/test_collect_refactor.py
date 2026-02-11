from __future__ import annotations

from typing import Any

from collections.abc import Callable
from pathlib import Path

import pytest

import aiaccel.hpo.modelbridge.collect as collect_module
from aiaccel.hpo.modelbridge.config import BridgeConfig, load_bridge_config
from aiaccel.hpo.modelbridge.toolkit.results import StepResult


def _config(tmp_path: Path, make_bridge_config: Callable[[str], dict[str, Any]]) -> BridgeConfig:
    payload = make_bridge_config(str(tmp_path / "outputs"))
    payload["hpo"]["base_config"] = str(tmp_path / "optimize.yaml")
    (tmp_path / "optimize.yaml").write_text("optimize:\n  goal: minimize\n", encoding="utf-8")
    return load_bridge_config(payload)


def test_collect_train_uses_finalize_helper(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, make_bridge_config)
    observed: dict[str, Any] = {"called": False}

    monkeypatch.setattr(
        collect_module,
        "_collect_pairs_for_scenario",
        lambda **_kwargs: ([], "stubbed"),
    )

    def fake_finalize(
        *,
        output_dir: Path,
        step: str,
        strict_mode: bool,
        scenario_outputs: dict[str, dict[str, Any]],
        issues: list[str],
        inputs: dict[str, Any] | None = None,
        extra_outputs: dict[str, Any] | None = None,
    ) -> StepResult:
        observed["called"] = True
        observed["step"] = step
        observed["strict"] = strict_mode
        _ = output_dir
        _ = scenario_outputs
        _ = issues
        _ = extra_outputs
        return StepResult(
            step=step,
            status="skipped",
            inputs=inputs or {},
            outputs={"scenarios": {}},
            reason="stubbed",
        )

    monkeypatch.setattr(collect_module, "finalize_scenario_step", fake_finalize)
    result = collect_module.collect_train(config)

    assert observed["called"] is True
    assert observed["step"] == "collect_train"
    assert result.status == "skipped"
