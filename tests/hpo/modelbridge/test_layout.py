from __future__ import annotations

from pathlib import Path

from aiaccel.hpo.modelbridge import layout


def test_layout_paths(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    sc_dir = layout.scenario_dir(output_dir, "demo")

    assert sc_dir == output_dir / "demo"
    assert layout.runs_dir(sc_dir, "train") == output_dir / "demo" / "runs" / "train"
    assert layout.run_dir(sc_dir, "train", 0, "macro") == output_dir / "demo" / "runs" / "train" / "000" / "macro"
    assert layout.train_run_dir(sc_dir, 1, "micro") == output_dir / "demo" / "runs" / "train" / "001" / "micro"
    assert layout.eval_run_dir(sc_dir, 2, "macro") == output_dir / "demo" / "runs" / "eval" / "002" / "macro"
    assert layout.models_dir(sc_dir) == output_dir / "demo" / "models"
    assert layout.metrics_dir(sc_dir) == output_dir / "demo" / "metrics"
    assert layout.workspace_dir(output_dir) == output_dir / "workspace"
    assert layout.state_dir(output_dir) == output_dir / "workspace" / "state"
    assert layout.commands_dir(output_dir) == output_dir / "workspace" / "commands"
    assert layout.logs_dir(output_dir) == output_dir / "workspace" / "logs"
    assert layout.optimize_logs_dir(output_dir) == output_dir / "workspace" / "logs" / "optimize"
    assert layout.train_plan_path(output_dir) == output_dir / "workspace" / "train_plan.json"
    assert layout.eval_plan_path(output_dir) == output_dir / "workspace" / "eval_plan.json"
    assert layout.state_path(output_dir, "prepare_train") == output_dir / "workspace" / "state" / "prepare_train.json"
    assert layout.command_path(output_dir, "train", "shell") == output_dir / "workspace" / "commands" / "train.sh"
    assert layout.command_path(output_dir, "eval", "json") == output_dir / "workspace" / "commands" / "eval.json"
    assert layout.optimize_log_path(output_dir, "train", "demo", 3, "micro") == (
        output_dir / "workspace" / "logs" / "optimize" / "train-demo-003-micro.log"
    )
