from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from aiaccel.hpo.modelbridge.common import (
    StepResult,
    hash_file,
    read_csv,
    read_json,
    resolve_seed,
    write_csv,
    write_json,
    write_step_state,
)
from aiaccel.hpo.modelbridge.config import SeedPolicyConfig, SeedUserValues


def test_toolkit_package_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("aiaccel.hpo.modelbridge.toolkit")


def test_common_io_roundtrip(tmp_path: Path) -> None:
    json_path = write_json(tmp_path / "payload.json", {"value": 123})
    assert read_json(json_path) == {"value": 123}
    assert len(hash_file(json_path)) == 64

    csv_path = write_csv(tmp_path / "pairs.csv", [{"run_id": 0, "macro_x": 1.0, "micro_y": 2.0}])
    rows = read_csv(csv_path)
    assert rows == [{"macro_x": "1.0", "micro_y": "2.0", "run_id": "0"}]


def test_step_state_writer(tmp_path: Path) -> None:
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
    assert payload["step"] == "prepare_train"
    assert payload["status"] == "success"
    assert payload["inputs"]["role"] == "train"
    assert payload["outputs"]["plan_path"] == "workspace/train_plan.json"


def test_step_result_requires_reason_for_failed_or_skipped() -> None:
    with pytest.raises(ValueError):
        StepResult(step="collect_train", status="failed")
    with pytest.raises(ValueError):
        StepResult(step="collect_train", status="skipped", reason="")
    StepResult(step="collect_train", status="success")


def test_common_seed_resolution() -> None:
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
