from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from aiaccel.hpo.modelbridge import common


def test_layout_module_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("aiaccel.hpo.modelbridge.layout")


def test_common_path_helpers(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    scenario_path = common.scenario_path(output_dir, "demo")

    assert scenario_path == output_dir / "demo"
    assert common.workspace_path(output_dir) == output_dir / "workspace"
    assert common.plan_path(output_dir, "train") == output_dir / "workspace" / "train_plan.json"
    assert common.plan_path(output_dir, "eval") == output_dir / "workspace" / "eval_plan.json"
    assert common.state_path(output_dir, "prepare_train") == output_dir / "workspace" / "state" / "prepare_train.json"
    assert common.command_path(output_dir, "train", "shell") == output_dir / "workspace" / "commands" / "train.sh"
    assert common.command_path(output_dir, "eval", "json") == output_dir / "workspace" / "commands" / "eval.json"
    assert common.optimize_log_path(output_dir, "train", "demo", 3, "micro") == (
        output_dir / "workspace" / "logs" / "optimize" / "train-demo-003-micro.log"
    )
    assert common.run_path(scenario_path, "train", 1, "macro") == (
        output_dir / "demo" / "runs" / "train" / "001" / "macro"
    )
