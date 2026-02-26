"""HPO execution steps for modelbridge."""

from __future__ import annotations

from typing import Any, cast

from collections.abc import Sequence
from pathlib import Path
import shlex
import subprocess

from .common import (
    Role,
    StepResult,
    StepStatus,
    command_path,
    plan_path,
    read_json,
    read_plan,
    write_json,
    write_step_state,
)
from .config import BridgeConfig
from .execution import emit_commands


def hpo_train(config: BridgeConfig) -> StepResult:
    """Execute train-role HPO commands.

    This step reads the train plan, renders run-level scripts, and executes the
    generated command bundle.

    Args:
        config: Validated modelbridge configuration.

    Returns:
        StepResult: Execution result for ``hpo_train``.

    Raises:
        ValueError: If plan or command artifacts are malformed.
        RuntimeError: If shell execution of generated commands fails.
    """
    return _run_role(config, role="train")


def hpo_eval(config: BridgeConfig) -> StepResult:
    """Execute eval-role HPO commands.

    This step reads the eval plan, renders run-level scripts, and executes the
    generated command bundle.

    Args:
        config: Validated modelbridge configuration.

    Returns:
        StepResult: Execution result for ``hpo_eval``.

    Raises:
        ValueError: If plan or command artifacts are malformed.
        RuntimeError: If shell execution of generated commands fails.
    """
    return _run_role(config, role="eval")


def _run_role(config: BridgeConfig, *, role: Role) -> StepResult:
    """Generate per-run scripts and execute one role command bundle."""
    output_dir = config.bridge.output_dir
    step_name = f"hpo_{role}"
    source_plan = plan_path(output_dir, role)
    if not source_plan.exists():
        result = StepResult(
            step=step_name,
            status="skipped",
            inputs={"role": role},
            outputs={},
            reason=f"Plan file not found: {source_plan}",
        )
        write_step_state(output_dir, result)
        return result

    plan_role, plan_entries = read_plan(source_plan)
    if plan_role != role:
        raise ValueError(f"Plan role mismatch: expected {role}, got {plan_role} ({source_plan})")
    if not plan_entries:
        result = StepResult(
            step=step_name,
            status="skipped",
            inputs={"role": role, "plan_path": str(source_plan)},
            outputs={"num_entries": 0},
            reason=f"No {role} plan entries",
        )
        write_step_state(output_dir, result)
        return result

    emitted_json_path = emit_commands(config, role=role, fmt="json")
    command_entries = _load_command_entries(emitted_json_path)

    run_scripts: list[Path] = []
    for entry in command_entries:
        run_scripts.append(_write_run_artifacts(entry))

    role_shell_path = _write_role_script(output_dir, role, run_scripts)
    role_json_path = _write_role_json(output_dir, role, command_entries, run_scripts)

    try:
        subprocess.run(["bash", str(role_shell_path)], check=True)
    except subprocess.CalledProcessError as exc:
        result = StepResult(
            step=step_name,
            status="failed",
            inputs={"role": role, "plan_path": str(source_plan)},
            outputs={
                "command_shell_path": str(role_shell_path),
                "command_json_path": str(role_json_path),
                "num_entries": len(plan_entries),
                "num_scripts": len(run_scripts),
            },
            reason=f"{step_name} execution failed with return code {exc.returncode}",
        )
        write_step_state(output_dir, result)
        raise RuntimeError(result.reason) from exc

    missing_dbs = [entry["expected_db_path"] for entry in plan_entries if not Path(entry["expected_db_path"]).exists()]
    status: StepStatus = "success" if not missing_dbs else "partial"
    reason = None if not missing_dbs else f"Missing expected DB files: {', '.join(missing_dbs[:3])}"
    if missing_dbs and len(missing_dbs) > 3:
        reason = f"{reason}, ... (+{len(missing_dbs) - 3} more)"

    result = StepResult(
        step=step_name,
        status=status,
        inputs={"role": role, "plan_path": str(source_plan)},
        outputs={
            "command_shell_path": str(role_shell_path),
            "command_json_path": str(role_json_path),
            "run_scripts": [str(path) for path in run_scripts],
            "num_entries": len(plan_entries),
            "missing_db_paths": missing_dbs,
        },
        reason=reason,
    )
    write_step_state(output_dir, result)
    return result


def _load_command_entries(path: Path) -> list[dict[str, Any]]:
    """Load emitted JSON command entries."""
    payload = read_json(path)
    commands = payload.get("commands") if isinstance(payload, dict) else None
    if not isinstance(commands, list):
        raise ValueError(f"Malformed command JSON: {path}")
    entries: list[dict[str, Any]] = []
    for index, item in enumerate(commands):
        if not isinstance(item, dict):
            raise ValueError(f"Malformed command entry: {path}#{index}")
        command = item.get("command")
        if not isinstance(command, list) or any(not isinstance(token, str) for token in command):
            raise ValueError(f"Malformed command list: {path}#{index}")
        entries.append(dict(item))
    return entries


def _write_run_artifacts(entry: dict[str, Any]) -> Path:
    """Write per-run executable script and JSON payload."""
    config_path_raw = entry.get("config_path")
    if not isinstance(config_path_raw, str) or not config_path_raw:
        raise ValueError("Malformed command entry: config_path is required")

    run_dir = Path(config_path_raw).expanduser().resolve().parent
    run_dir.mkdir(parents=True, exist_ok=True)

    command = cast(Sequence[str], entry["command"])
    run_script_path = run_dir / "run.sh"
    run_json_path = run_dir / "run.json"

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        " ".join(shlex.quote(token) for token in command),
        "",
    ]
    run_script_path.write_text("\n".join(lines), encoding="utf-8")
    run_script_path.chmod(0o755)

    write_json(
        run_json_path,
        {
            "scenario": entry.get("scenario"),
            "role": entry.get("role"),
            "run_id": entry.get("run_id"),
            "target": entry.get("target"),
            "config_path": config_path_raw,
            "execution_target": entry.get("execution_target"),
            "command": list(command),
        },
    )
    return run_script_path


def _write_role_script(output_dir: Path, role: Role, run_scripts: Sequence[Path]) -> Path:
    """Write role-level shell script that calls per-run scripts."""
    shell_path = command_path(output_dir, role, "shell")
    shell_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    lines.extend(f"bash {shlex.quote(str(path))}" for path in run_scripts)
    lines.append("")
    shell_path.write_text("\n".join(lines), encoding="utf-8")
    shell_path.chmod(0o755)
    return shell_path


def _write_role_json(
    output_dir: Path,
    role: Role,
    command_entries: Sequence[dict[str, Any]],
    run_scripts: Sequence[Path],
) -> Path:
    """Write role-level JSON command metadata."""
    json_path = command_path(output_dir, role, "json")
    write_json(
        json_path,
        {
            "role": role,
            "commands": list(command_entries),
            "run_scripts": [str(path) for path in run_scripts],
        },
    )
    return json_path
