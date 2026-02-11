"""Collect steps for modelbridge."""

from __future__ import annotations

from typing import Any

from collections.abc import Sequence
from pathlib import Path

from .config import BridgeConfig, ScenarioConfig
from .layout import Role, plan_path, scenario_dir
from .storage import load_pairs_from_db_pairs, scan_db_paths_for_pairs, scan_runs_for_pairs
from .toolkit.io import read_json, write_csv
from .toolkit.logging import get_logger
from .toolkit.results import StepResult, finalize_scenario_step

_logger = get_logger(__name__)


def collect_train(
    config: BridgeConfig,
    db_paths: Sequence[Path] | None = None,
    db_pairs: Sequence[tuple[Path, Path]] | None = None,
) -> StepResult:
    """Collect training macro/micro trial pairs into CSV artifacts.

    Args:
        config: Parsed modelbridge configuration.
        db_paths: Optional explicit database path list.
        db_pairs: Optional explicit `(macro_db, micro_db)` pairs.

    Returns:
        StepResult: Step execution result for `collect_train`.

    Raises:
        RuntimeError: When strict mode is enabled and no valid pairs are found.
    """
    return _collect_role(config, "train", db_paths=db_paths, db_pairs=db_pairs)


def collect_eval(
    config: BridgeConfig,
    db_paths: Sequence[Path] | None = None,
    db_pairs: Sequence[tuple[Path, Path]] | None = None,
) -> StepResult:
    """Collect evaluation macro/micro trial pairs into CSV artifacts.

    Args:
        config: Parsed modelbridge configuration.
        db_paths: Optional explicit database path list.
        db_pairs: Optional explicit `(macro_db, micro_db)` pairs.

    Returns:
        StepResult: Step execution result for `collect_eval`.

    Raises:
        RuntimeError: When strict mode is enabled and no valid pairs are found.
    """
    return _collect_role(config, "eval", db_paths=db_paths, db_pairs=db_pairs)


def _collect_role(
    config: BridgeConfig,
    role: Role,
    db_paths: Sequence[Path] | None = None,
    db_pairs: Sequence[tuple[Path, Path]] | None = None,
) -> StepResult:
    """Collect macro/micro pairs for one role and persist state."""
    output_dir = config.bridge.output_dir
    scenario_outputs: dict[str, dict[str, Any]] = {}
    issues: list[str] = []

    for scenario in config.bridge.scenarios:
        scenario_path = scenario_dir(output_dir, scenario.name)
        scenario_path.mkdir(parents=True, exist_ok=True)

        selected_pairs, source_name = _collect_pairs_for_scenario(
            config=config,
            scenario=scenario,
            scenario_path=scenario_path,
            role=role,
            db_paths=db_paths,
            db_pairs=db_pairs,
        )

        csv_path = scenario_path / ("train_pairs.csv" if role == "train" else "test_pairs.csv")
        if selected_pairs:
            rows = _pairs_to_rows(selected_pairs)
            write_csv(csv_path, rows)
            scenario_outputs[scenario.name] = {
                "status": "success",
                "pairs_csv": str(csv_path),
                "num_pairs": len(selected_pairs),
                "source": source_name,
            }
            continue

        scenario_outputs[scenario.name] = {
            "status": "missing",
            "pairs_csv": str(csv_path),
            "num_pairs": 0,
            "source": source_name,
        }
        issues.append(f"{scenario.name}:{role} has no valid macro/micro pairs")

    return finalize_scenario_step(
        output_dir=output_dir,
        step=f"collect_{role}",
        strict_mode=config.bridge.strict_mode,
        scenario_outputs=scenario_outputs,
        issues=issues,
        inputs=_collect_inputs(role, db_paths, db_pairs),
    )


def _collect_inputs(
    role: Role,
    db_paths: Sequence[Path] | None,
    db_pairs: Sequence[tuple[Path, Path]] | None,
) -> dict[str, Any]:
    """Build normalized input payload for step state output."""
    return {
        "role": role,
        "db_paths": [str(path) for path in db_paths or []],
        "db_pairs": [[str(macro), str(micro)] for macro, micro in db_pairs or []],
    }


def _collect_pairs_for_scenario(
    config: BridgeConfig,
    scenario: ScenarioConfig,
    scenario_path: Path,
    role: Role,
    db_paths: Sequence[Path] | None,
    db_pairs: Sequence[tuple[Path, Path]] | None,
) -> tuple[list[tuple[int, dict[str, float], dict[str, float]]], str]:
    """Resolve pair sources with explicit inputs first and fallbacks next."""
    if db_pairs:
        return load_pairs_from_db_pairs(scenario, role, db_pairs), "explicit_db_pairs"

    if db_paths:
        return scan_db_paths_for_pairs(scenario, role, db_paths), "explicit_db_paths"

    plan_paths = _load_db_paths_from_plan(config.bridge.output_dir, scenario.name, role)
    if plan_paths:
        pairs = scan_db_paths_for_pairs(scenario, role, plan_paths)
        if pairs:
            return pairs, "plan"

    return scan_runs_for_pairs(scenario, scenario_path, role), "layout_scan"


def _load_db_paths_from_plan(output_dir: Path, scenario_name: str, role: Role) -> list[Path]:
    """Load expected DB paths for a scenario from role plan."""
    path = plan_path(output_dir, role)
    if not path.exists():
        return []
    payload = read_json(path)
    entries = payload.get("entries", []) if isinstance(payload, dict) else []
    db_paths: list[Path] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("scenario") != scenario_name:
            continue
        db_path_value = entry.get("expected_db_path")
        if not isinstance(db_path_value, str):
            continue
        db_path = Path(db_path_value)
        if db_path.exists():
            db_paths.append(db_path)
    return db_paths


def _pairs_to_rows(pairs: Sequence[tuple[int, dict[str, float], dict[str, float]]]) -> list[dict[str, Any]]:
    """Convert pair tuples into CSV row dictionaries."""
    rows: list[dict[str, Any]] = []
    for run_id, macro, micro in pairs:
        row: dict[str, Any] = {"run_id": run_id}
        row.update({f"macro_{name}": value for name, value in macro.items()})
        row.update({f"micro_{name}": value for name, value in micro.items()})
        rows.append(row)
    return rows
