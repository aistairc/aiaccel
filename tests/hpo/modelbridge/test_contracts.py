from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from aiaccel.hpo.modelbridge.common import read_plan, write_json


def _entry() -> dict[str, object]:
    return {
        "scenario": "s",
        "role": "train",
        "run_id": 0,
        "target": "macro",
        "config_path": "/tmp/config.yaml",
        "expected_db_path": "/tmp/optuna.db",
        "study_name": "s-train-macro-000",
        "seed": 1,
        "sampler_seed": 1,
        "optimizer_seed": 2,
        "seed_mode": "auto_increment",
        "sampler_seed_mode": "auto_increment",
        "optimizer_seed_mode": "auto_increment",
        "execution_target": "local",
        "objective_command": ["python", "obj.py"],
    }


def test_contracts_module_removed() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("aiaccel.hpo.modelbridge.contracts")


def test_read_plan_round_trip(tmp_path: Path) -> None:
    plan_path = tmp_path / "workspace" / "train_plan.json"
    write_json(plan_path, {"role": "train", "created_at": "2026-02-11T00:00:00+00:00", "entries": [_entry()]})

    role, entries = read_plan(plan_path)
    assert role == "train"
    assert len(entries) == 1
    assert entries[0]["study_name"] == "s-train-macro-000"


def test_read_plan_requires_role(tmp_path: Path) -> None:
    plan_path = tmp_path / "workspace" / "train_plan.json"
    payload = {"role": "train", "created_at": "2026-02-11T00:00:00+00:00", "entries": [_entry()]}
    payload.pop("role")
    write_json(plan_path, payload)

    with pytest.raises(ValueError, match="role"):
        read_plan(plan_path)


def test_read_plan_rejects_negative_run_id(tmp_path: Path) -> None:
    plan_path = tmp_path / "workspace" / "train_plan.json"
    bad = _entry()
    bad["run_id"] = -1
    write_json(plan_path, {"role": "train", "created_at": "2026-02-11T00:00:00+00:00", "entries": [bad]})

    with pytest.raises(ValueError, match="run_id"):
        read_plan(plan_path)
