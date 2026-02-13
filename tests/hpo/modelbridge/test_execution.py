from __future__ import annotations

from typing import Any

from collections.abc import Callable
from pathlib import Path

import pytest

from aiaccel.hpo.modelbridge.common import read_json, write_json
from aiaccel.hpo.modelbridge.config import load_bridge_config
from aiaccel.hpo.modelbridge.execution import emit_commands
from aiaccel.hpo.modelbridge.prepare import prepare_train


def _config(tmp_path: Path, make_bridge_config: Callable[[str], dict[str, Any]]) -> Any:
    payload = make_bridge_config(str(tmp_path / "outputs"))
    payload["hpo"]["base_config"] = str(tmp_path / "optimize.yaml")
    (tmp_path / "optimize.yaml").write_text("optimize:\n  goal: minimize\n", encoding="utf-8")
    return load_bridge_config(payload)


def test_emit_commands_deterministic(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
) -> None:
    config = _config(tmp_path, make_bridge_config)
    prepare_train(config)

    path1 = emit_commands(config, role="train", fmt="json")
    path2 = emit_commands(config, role="train", fmt="json")

    assert path1 == path2
    assert read_json(path1) == read_json(path2)

    shell_path = emit_commands(config, role="train", fmt="shell")
    content = shell_path.read_text(encoding="utf-8")
    assert "aiaccel-hpo optimize --config" in content


def test_emit_commands_requires_plan(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
) -> None:
    config = _config(tmp_path, make_bridge_config)
    with pytest.raises(FileNotFoundError):
        emit_commands(config, role="train", fmt="json")


def test_emit_commands_abci_wrapper(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
) -> None:
    payload = make_bridge_config(str(tmp_path / "outputs"))
    payload["hpo"]["base_config"] = str(tmp_path / "optimize.yaml")
    payload["bridge"]["execution"] = {
        "target": "abci",
        "job_profile": "sge",
        "job_mode": "cpu",
        "job_walltime": "1:00:00",
        "job_extra_args": ["--n_tasks", "2"],
    }
    (tmp_path / "optimize.yaml").write_text("optimize:\n  goal: minimize\n", encoding="utf-8")
    config = load_bridge_config(payload)
    prepare_train(config)

    shell_path = emit_commands(config, role="train", fmt="shell")
    content = shell_path.read_text(encoding="utf-8")
    assert "aiaccel-job sge cpu --walltime 1:00:00 --n_tasks 2" in content
    assert "-- aiaccel-hpo optimize --config" in content
    assert (config.bridge.output_dir / "workspace" / "logs" / "optimize").exists()


def test_emit_commands_target_override(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
) -> None:
    payload = make_bridge_config(str(tmp_path / "outputs"))
    payload["hpo"]["base_config"] = str(tmp_path / "optimize.yaml")
    payload["bridge"]["execution"] = {"target": "abci"}
    (tmp_path / "optimize.yaml").write_text("optimize:\n  goal: minimize\n", encoding="utf-8")
    config = load_bridge_config(payload)
    prepare_train(config)

    shell_path = emit_commands(config, role="train", fmt="shell", execution_target="local")
    content = shell_path.read_text(encoding="utf-8")
    assert "aiaccel-job" not in content
    assert "aiaccel-hpo optimize --config" in content


def test_emit_commands_target_override_local_to_abci_defaults(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
) -> None:
    config = _config(tmp_path, make_bridge_config)
    prepare_train(config)

    shell_path = emit_commands(config, role="train", fmt="shell", execution_target="abci")
    content = shell_path.read_text(encoding="utf-8")
    assert "aiaccel-job sge cpu" in content
    assert "aiaccel-job local" not in content


def test_emit_commands_uses_sort_helper(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path, make_bridge_config)
    prepare_train(config)
    called = {"value": False}

    def fake_sort(entries: list[Any]) -> list[Any]:
        called["value"] = True
        assert entries
        return entries

    monkeypatch.setattr("aiaccel.hpo.modelbridge.execution._sort_command_entries", fake_sort)
    emit_commands(config, role="train", fmt="json")
    assert called["value"] is True


def test_emit_commands_rejects_plan_entry_without_role(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
) -> None:
    config = _config(tmp_path, make_bridge_config)
    prepare_train(config)
    plan_path = config.bridge.output_dir / "workspace" / "train_plan.json"
    plan = read_json(plan_path)
    entries = plan["entries"]
    assert entries
    entries[0].pop("role", None)
    write_json(plan_path, plan)

    with pytest.raises(ValueError, match="role"):
        emit_commands(config, role="train", fmt="json")


def test_emit_commands_abci_uses_optimize_log_path_helper(
    tmp_path: Path,
    make_bridge_config: Callable[[str], dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = make_bridge_config(str(tmp_path / "outputs"))
    payload["hpo"]["base_config"] = str(tmp_path / "optimize.yaml")
    payload["bridge"]["execution"] = {"target": "abci"}
    (tmp_path / "optimize.yaml").write_text("optimize:\n  goal: minimize\n", encoding="utf-8")
    config = load_bridge_config(payload)
    prepare_train(config)

    observed: dict[str, object] = {}

    def fake_optimize_log_path(output_dir: Path, role: str, scenario: str, run_id: int, target: str) -> Path:
        observed["args"] = (output_dir, role, scenario, run_id, target)
        return output_dir / "workspace" / "logs" / "optimize" / "custom-log-name.log"

    monkeypatch.setattr("aiaccel.hpo.modelbridge.execution.optimize_log_path", fake_optimize_log_path)
    shell_path = emit_commands(config, role="train", fmt="shell")

    content = shell_path.read_text(encoding="utf-8")
    assert "custom-log-name.log" in content
    assert observed["args"]
