from __future__ import annotations

from pathlib import Path

from aiaccel.hpo.modelbridge.toolkit.command_render import render_json_commands, render_shell_commands
from aiaccel.hpo.modelbridge.toolkit.io import hash_file, read_csv, read_json, write_csv, write_json
from aiaccel.hpo.modelbridge.toolkit.results import StepResult, write_step_state


def test_toolkit_io_roundtrip(tmp_path: Path) -> None:
    json_path = write_json(tmp_path / "payload.json", {"value": 123})
    assert read_json(json_path) == {"value": 123}
    assert len(hash_file(json_path)) == 64

    csv_path = write_csv(tmp_path / "pairs.csv", [{"run_id": 0, "macro_x": 1.0, "micro_y": 2.0}])
    rows = read_csv(csv_path)
    assert rows == [{"macro_x": "1.0", "micro_y": "2.0", "run_id": "0"}]


def test_toolkit_renderers() -> None:
    commands = [{"scenario": "demo", "command": ["aiaccel-hpo", "optimize", "--config", "conf.yaml"]}]
    payload = render_json_commands("train", commands)
    assert payload["role"] == "train"
    assert payload["commands"][0]["scenario"] == "demo"

    shell = render_shell_commands([["echo", "hello world"]])
    assert shell.startswith("#!/usr/bin/env bash")
    assert "echo 'hello world'" in shell


def test_toolkit_step_state_writer(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    result = StepResult(
        step="prepare_train",
        status="success",
        inputs={"role": "train"},
        outputs={"plan_path": "workspace/train_plan.json"},
    )
    state_path = write_step_state(output_dir, result)

    assert state_path == output_dir / "workspace" / "state" / "prepare_train.json"
    payload = read_json(state_path)
    assert payload["status"] == "success"
    assert payload["inputs"]["role"] == "train"
    assert payload["outputs"]["plan_path"] == "workspace/train_plan.json"
