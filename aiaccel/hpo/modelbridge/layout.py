"""Directory layout helpers for modelbridge."""

from __future__ import annotations

from typing import Literal

from pathlib import Path

Role = Literal["train", "eval"]
Target = Literal["macro", "micro"]


def scenario_dir(output_dir: Path, scenario_name: str) -> Path:
    """Return scenario directory path.

    Args:
        output_dir: Root output directory.
        scenario_name: Scenario name.

    Returns:
        Path: Scenario directory path.
    """
    return output_dir / scenario_name


def runs_dir(scenario_dir_path: Path, role: Role) -> Path:
    """Return runs directory for a role.

    Args:
        scenario_dir_path: Scenario directory path.
        role: Target role.

    Returns:
        Path: Runs directory path.
    """
    return scenario_dir_path / "runs" / role


def run_dir(scenario_dir_path: Path, role: Role, run_idx: int, target: Target) -> Path:
    """Return directory for one run/target.

    Args:
        scenario_dir_path: Scenario directory path.
        role: Target role.
        run_idx: Run index.
        target: Target kind (`macro` or `micro`).

    Returns:
        Path: Run directory path.
    """
    return runs_dir(scenario_dir_path, role) / f"{run_idx:03d}" / target


def train_run_dir(scenario_dir_path: Path, run_idx: int, target: Target) -> Path:
    """Return train run directory path.

    Args:
        scenario_dir_path: Scenario directory path.
        run_idx: Run index.
        target: Target kind.

    Returns:
        Path: Train run directory path.
    """
    return run_dir(scenario_dir_path, "train", run_idx, target)


def eval_run_dir(scenario_dir_path: Path, run_idx: int, target: Target) -> Path:
    """Return evaluation run directory path.

    Args:
        scenario_dir_path: Scenario directory path.
        run_idx: Run index.
        target: Target kind.

    Returns:
        Path: Eval run directory path.
    """
    return run_dir(scenario_dir_path, "eval", run_idx, target)


def models_dir(scenario_dir_path: Path) -> Path:
    """Return models directory path.

    Args:
        scenario_dir_path: Scenario directory path.

    Returns:
        Path: Models directory path.
    """
    return scenario_dir_path / "models"


def metrics_dir(scenario_dir_path: Path) -> Path:
    """Return metrics directory path.

    Args:
        scenario_dir_path: Scenario directory path.

    Returns:
        Path: Metrics directory path.
    """
    return scenario_dir_path / "metrics"


def workspace_dir(output_dir: Path) -> Path:
    """Return workspace directory path.

    Args:
        output_dir: Root output directory.

    Returns:
        Path: Workspace directory path.
    """
    return output_dir / "workspace"


def state_dir(output_dir: Path) -> Path:
    """Return state directory path.

    Args:
        output_dir: Root output directory.

    Returns:
        Path: State directory path.
    """
    return workspace_dir(output_dir) / "state"


def commands_dir(output_dir: Path) -> Path:
    """Return command directory path.

    Args:
        output_dir: Root output directory.

    Returns:
        Path: Command directory path.
    """
    return workspace_dir(output_dir) / "commands"


def logs_dir(output_dir: Path) -> Path:
    """Return workspace logs directory path.

    Args:
        output_dir: Root output directory.

    Returns:
        Path: Workspace logs directory path.
    """
    return workspace_dir(output_dir) / "logs"


def optimize_logs_dir(output_dir: Path) -> Path:
    """Return optimize log directory path.

    Args:
        output_dir: Root output directory.

    Returns:
        Path: Optimize log directory path.
    """
    return logs_dir(output_dir) / "optimize"


def optimize_log_path(output_dir: Path, role: Role, scenario: str, run_id: int, target: Target) -> Path:
    """Return optimize log path for one plan entry.

    Args:
        output_dir: Root output directory.
        role: Role (`train` or `eval`).
        scenario: Scenario name.
        run_id: Run index.
        target: Target (`macro` or `micro`).

    Returns:
        Path: Log file path for wrapped optimize execution.
    """
    return optimize_logs_dir(output_dir) / f"{role}-{scenario}-{run_id:03d}-{target}.log"


def train_plan_path(output_dir: Path) -> Path:
    """Return train plan file path.

    Args:
        output_dir: Root output directory.

    Returns:
        Path: Train plan path.
    """
    return workspace_dir(output_dir) / "train_plan.json"


def eval_plan_path(output_dir: Path) -> Path:
    """Return eval plan file path.

    Args:
        output_dir: Root output directory.

    Returns:
        Path: Eval plan path.
    """
    return workspace_dir(output_dir) / "eval_plan.json"


def plan_path(output_dir: Path, role: Role) -> Path:
    """Return plan file path for role.

    Args:
        output_dir: Root output directory.
        role: Target role.

    Returns:
        Path: Role-specific plan path.
    """
    if role == "train":
        return train_plan_path(output_dir)
    return eval_plan_path(output_dir)


def state_path(output_dir: Path, step_name: str) -> Path:
    """Return state JSON path for a step.

    Args:
        output_dir: Root output directory.
        step_name: Step name.

    Returns:
        Path: State file path.
    """
    return state_dir(output_dir) / f"{step_name}.json"


def command_path(output_dir: Path, role: Role, fmt: Literal["shell", "json"]) -> Path:
    """Return emitted command file path.

    Args:
        output_dir: Root output directory.
        role: Target role.
        fmt: Output format.

    Returns:
        Path: Command artifact path.
    """
    suffix = "sh" if fmt == "shell" else "json"
    return commands_dir(output_dir) / f"{role}.{suffix}"
