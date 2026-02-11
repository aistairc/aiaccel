from __future__ import annotations

from pathlib import Path

import pytest

from aiaccel.hpo.modelbridge.config import ExecutionTargetConfig, SeedPolicyConfig, SeedUserValues
from aiaccel.hpo.modelbridge.toolkit.command_render import (
    render_json_commands,
    render_shell_commands,
    sort_command_entries,
)
from aiaccel.hpo.modelbridge.toolkit.io import hash_file, read_csv, read_json, write_csv, write_json
from aiaccel.hpo.modelbridge.toolkit.job_command import wrap_with_aiaccel_job
from aiaccel.hpo.modelbridge.toolkit.results import StepResult, write_step_state
from aiaccel.hpo.modelbridge.toolkit.seeding import resolve_seed


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


def test_toolkit_sort_command_entries() -> None:
    entries = [
        {"scenario": "demo", "run_id": 3, "target": "micro"},
        {"scenario": "demo", "run_id": 1, "target": "micro"},
        {"scenario": "alpha", "run_id": 2, "target": "macro"},
    ]
    sorted_entries = sort_command_entries(entries)
    assert [(item["scenario"], item["run_id"], item["target"]) for item in sorted_entries] == [
        ("alpha", 2, "macro"),
        ("demo", 1, "micro"),
        ("demo", 3, "micro"),
    ]


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


def test_step_result_requires_reason_for_failed_or_skipped() -> None:
    with pytest.raises(ValueError):
        StepResult(step="collect_train", status="failed")
    with pytest.raises(ValueError):
        StepResult(step="collect_train", status="skipped", reason="")
    StepResult(step="collect_train", status="success")


def test_toolkit_seed_resolution() -> None:
    auto_policy = SeedPolicyConfig(mode="auto_increment", base=10)
    assert resolve_seed(auto_policy, role="train", target="macro", run_id=0, fallback_base=5) == 10
    assert resolve_seed(auto_policy, role="eval", target="micro", run_id=2, fallback_base=5) == 300012

    user_policy = SeedPolicyConfig(
        mode="user_defined",
        user_values=SeedUserValues(
            train_macro=[101],
            train_micro=[201],
            eval_macro=[301],
            eval_micro=[401],
        ),
    )
    assert resolve_seed(user_policy, role="eval", target="micro", run_id=0, fallback_base=0) == 401
    with pytest.raises(ValueError):
        resolve_seed(user_policy, role="eval", target="micro", run_id=2, fallback_base=0)


def test_toolkit_wrap_with_aiaccel_job() -> None:
    execution = ExecutionTargetConfig(
        target="abci",
        job_profile="sge",
        job_mode="cpu",
        job_walltime="1:00:00",
        job_extra_args=["--n_tasks", "4"],
    )
    wrapped = wrap_with_aiaccel_job(
        ["aiaccel-hpo", "optimize", "--config", "config.yaml"],
        execution,
        Path("logs/job.log"),
    )
    assert wrapped[:4] == ["aiaccel-job", "sge", "cpu", "--walltime"]
    assert "--n_tasks" in wrapped
    assert wrapped.index("--n_tasks") < wrapped.index("logs/job.log")
    assert wrapped.index("logs/job.log") < wrapped.index("--")
    assert "logs/job.log" in wrapped
    assert wrapped[-4:] == ["aiaccel-hpo", "optimize", "--config", "config.yaml"]
