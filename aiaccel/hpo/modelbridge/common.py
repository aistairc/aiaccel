"""Shared contracts and lightweight utilities for modelbridge."""

from __future__ import annotations

from typing import Any, Literal, cast

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from .config import SeedPolicyConfig

Role = Literal["train", "eval"]
Target = Literal["macro", "micro"]
CommandFormat = Literal["shell", "json"]
StepStatus = Literal["success", "skipped", "partial", "failed"]

_GROUP_INDEX: dict[tuple[Role, Target], int] = {
    ("train", "macro"): 0,
    ("train", "micro"): 1,
    ("eval", "macro"): 2,
    ("eval", "micro"): 3,
}
_SEED_KEY: dict[tuple[Role, Target], Literal["train_macro", "train_micro", "eval_macro", "eval_micro"]] = {
    ("train", "macro"): "train_macro",
    ("train", "micro"): "train_micro",
    ("eval", "macro"): "eval_macro",
    ("eval", "micro"): "eval_micro",
}
_OFFSET = 100000
_LOG_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_PLAN_STRING_FIELDS = (
    "scenario",
    "role",
    "target",
    "config_path",
    "expected_db_path",
    "study_name",
    "execution_target",
)
_PLAN_INT_FIELDS = ("run_id", "seed", "sampler_seed", "optimizer_seed")
_PLAN_MODE_FIELDS = ("seed_mode", "sampler_seed_mode", "optimizer_seed_mode")
_PLAN_MODES = {"auto_increment", "user_defined"}


@dataclass(frozen=True)
class StepResult:
    """Normalized per-step result payload.

    Args:
        step: Step name.
        status: Step execution status.
        inputs: Input payload summary.
        outputs: Output payload summary.
        reason: Optional reason for non-success statuses.
        timestamp: RFC3339 timestamp.
    """

    step: str
    status: StepStatus
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    reason: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        """Validate reason requirement for failed/skipped results."""
        if self.status in {"failed", "skipped"} and (self.reason is None or not self.reason.strip()):
            raise ValueError("reason is required when status is failed or skipped")

    def to_state(self) -> dict[str, Any]:
        """Return JSON-serializable state payload.

        Returns:
            dict[str, Any]: Persistable state mapping.
        """
        payload: dict[str, Any] = {
            "step": self.step,
            "status": self.status,
            "timestamp": self.timestamp,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }
        if self.reason is not None:
            payload["reason"] = self.reason
        return payload


@dataclass(frozen=True)
class PipelineResult:
    """Aggregate result for one pipeline invocation.

    Args:
        results: Ordered step results.
        summary_path: Optional summary JSON path.
        manifest_path: Optional manifest JSON path.
    """

    results: list[StepResult]
    summary_path: Path | None = None
    manifest_path: Path | None = None


def workspace_path(output_dir: Path) -> Path:
    """Return `workspace` directory path under output root."""
    return output_dir / "workspace"


def scenario_path(output_dir: Path, scenario: str) -> Path:
    """Return scenario output directory path."""
    return output_dir / scenario


def run_path(scenario_dir: Path, role: Role, run_id: int, target: Target) -> Path:
    """Return per-run/target directory path for one scenario."""
    return scenario_dir / "runs" / role / f"{run_id:03d}" / target


def plan_path(output_dir: Path, role: Role) -> Path:
    """Return role plan file path."""
    return workspace_path(output_dir) / f"{role}_plan.json"


def state_path(output_dir: Path, step: str) -> Path:
    """Return per-step state file path."""
    return workspace_path(output_dir) / "state" / f"{step}.json"


def command_path(output_dir: Path, role: Role, fmt: CommandFormat) -> Path:
    """Return command artifact path for one role and format."""
    suffix = "sh" if fmt == "shell" else "json"
    return workspace_path(output_dir) / "commands" / f"{role}.{suffix}"


def optimize_log_path(output_dir: Path, role: Role, scenario: str, run_id: int, target: Target) -> Path:
    """Return optimize log path for one role/scenario/run/target."""
    return workspace_path(output_dir) / "logs" / "optimize" / f"{role}-{scenario}-{run_id:03d}-{target}.log"


def setup_logging(log_level: str, output_dir: Path, *, json_logs: bool = False) -> Path:
    """Configure root logging handlers for modelbridge runtime."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "aiaccel_modelbridge.log"

    root = logging.getLogger()
    for handler in list(root.handlers):
        handler.close()
        root.removeHandler(handler)

    fmt = _LOG_FMT
    if json_logs:
        fmt = '{"time":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","msg":"%(message)s"}'
    formatter = logging.Formatter(fmt)

    if not os.environ.get("AIACCEL_LOG_NO_FILE"):
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    if not os.environ.get("AIACCEL_LOG_SILENT"):
        rich_handler = RichHandler(console=Console(), rich_tracebacks=True, omit_repeated_times=False, show_time=False)
        rich_handler.setFormatter(formatter)
        root.addHandler(rich_handler)

    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logging.captureWarnings(True)
    return log_path


def write_json(path: Path, payload: Any) -> Path:
    """Write JSON payload with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return path


def read_json(path: Path) -> Any:
    """Read UTF-8 JSON file and return decoded payload."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    """Write row mappings as CSV with sorted header keys."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    header = sorted(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in header})
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read CSV file as list of row dictionaries."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def hash_file(path: Path, algorithm: str = "sha256") -> str:
    """Return file hex digest."""
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve_seed(policy: SeedPolicyConfig, *, role: Role, target: Target, run_id: int, fallback_base: int) -> int:
    """Resolve deterministic seed for one role/target/run."""
    if policy.mode == "auto_increment":
        base = fallback_base if policy.base is None else policy.base
        return base + _GROUP_INDEX[(role, target)] * _OFFSET + run_id
    if policy.user_values is None:
        raise ValueError("user_values must be provided when mode=user_defined")
    values = list(getattr(policy.user_values, _SEED_KEY[(role, target)]))
    if run_id < 0 or run_id >= len(values):
        raise ValueError(f"Missing user-defined seed for {role}/{target} run_id={run_id}")
    return int(values[run_id])


def read_plan(path: Path) -> tuple[Role, list[dict[str, Any]]]:
    """Read and validate one role plan file."""
    payload = read_json(path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Malformed plan payload: {path}")

    role_value = payload.get("role")
    if role_value not in {"train", "eval"}:
        raise ValueError(f"Malformed plan role: {path}")
    role = cast(Role, role_value)

    entries_raw = payload.get("entries")
    if not isinstance(entries_raw, list):
        raise ValueError(f"Malformed plan entries: {path}")

    entries: list[dict[str, Any]] = []
    for index, raw in enumerate(entries_raw):
        if not isinstance(raw, Mapping):
            raise ValueError(f"Malformed plan entry: {path}#{index}")
        entries.append(_normalize_plan_entry(raw, source=f"{path}#{index}"))
    return role, entries


def _normalize_plan_entry(entry: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    """Normalize one raw plan entry mapping."""
    normalized: dict[str, Any] = {}
    for key in _PLAN_STRING_FIELDS:
        normalized[key] = _require_non_empty_str(entry, key, source)
    for key in _PLAN_INT_FIELDS:
        normalized[key] = _require_int(entry, key, source, non_negative=(key == "run_id"))
    for key in _PLAN_MODE_FIELDS:
        value = entry.get(key)
        if value not in _PLAN_MODES:
            raise ValueError(f"Malformed plan entry field '{key}': {source}")
        normalized[key] = value

    command = entry.get("objective_command")
    if not isinstance(command, list) or any(not isinstance(item, str) for item in command):
        raise ValueError(f"Malformed plan entry field 'objective_command': {source}")
    normalized["objective_command"] = list(command)

    if normalized["role"] not in {"train", "eval"}:
        raise ValueError(f"Malformed plan entry role: {source}")
    if normalized["target"] not in {"macro", "micro"}:
        raise ValueError(f"Malformed plan entry target: {source}")
    if normalized["execution_target"] not in {"local", "abci"}:
        raise ValueError(f"Malformed plan entry execution_target: {source}")
    return normalized


def _require_non_empty_str(entry: Mapping[str, Any], key: str, source: str) -> str:
    """Validate and return one non-empty string field."""
    value = entry.get(key)
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"Malformed plan entry field '{key}': {source}")


def _require_int(entry: Mapping[str, Any], key: str, source: str, *, non_negative: bool) -> int:
    """Validate and return one integer field."""
    value = entry.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Malformed plan entry field '{key}': {source}")
    if non_negative and value < 0:
        raise ValueError(f"Malformed plan entry field '{key}': {source}")
    return value


def write_step_state(output_dir: Path, result: StepResult) -> Path:
    """Persist one step result payload under workspace state."""
    return write_json(state_path(output_dir, result.step), result.to_state())


def finalize_scenario_step(
    *,
    output_dir: Path,
    step: str,
    strict_mode: bool,
    scenario_outputs: Mapping[str, Mapping[str, Any]],
    issues: Sequence[str],
    inputs: Mapping[str, Any] | None = None,
    extra_outputs: Mapping[str, Any] | None = None,
) -> StepResult:
    """Finalize per-scenario status and persist step state."""
    outputs: dict[str, Any] = {"scenarios": {name: dict(payload) for name, payload in scenario_outputs.items()}}
    if extra_outputs:
        outputs.update(dict(extra_outputs))

    success_count = sum(1 for payload in outputs["scenarios"].values() if payload.get("status") == "success")
    issue_list = list(issues)
    issue_text = "; ".join(issue_list)

    if issue_list and strict_mode:
        failed = StepResult(step=step, status="failed", inputs=dict(inputs or {}), outputs=outputs, reason=issue_text)
        write_step_state(output_dir, failed)
        raise RuntimeError(issue_text)

    reason: str | None = None
    if success_count == len(outputs["scenarios"]):
        status: StepStatus = "success"
    elif success_count == 0:
        status = "skipped"
        reason = issue_text or f"{step} skipped"
    else:
        status = "partial"
        reason = issue_text

    result = StepResult(step=step, status=status, inputs=dict(inputs or {}), outputs=outputs, reason=reason)
    write_step_state(output_dir, result)
    return result
