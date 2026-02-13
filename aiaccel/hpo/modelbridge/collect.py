"""Collect macro/micro best-parameter pairs from Optuna DB files."""

from __future__ import annotations

from typing import Any, cast

from collections.abc import Sequence
from pathlib import Path
import re

import optuna

from .common import Role, StepResult, Target, finalize_scenario_step, plan_path, read_plan, scenario_path, write_csv
from .config import BridgeConfig

Pair = tuple[int, dict[str, float], dict[str, float]]


def collect_train(
    config: BridgeConfig,
    db_paths: Sequence[Path] | None = None,
    db_pairs: Sequence[tuple[Path, Path]] | None = None,
) -> StepResult:
    """Collect train-role macro/micro best-parameter pairs.

    Args:
        config: Validated modelbridge configuration.
        db_paths: Optional explicit DB path list.
        db_pairs: Optional explicit macro/micro DB pair list.

    Returns:
        StepResult: Persisted step result for ``collect_train``.
    """
    return _collect_role(config, role="train", db_paths=tuple(db_paths or ()), db_pairs=tuple(db_pairs or ()))


def collect_eval(
    config: BridgeConfig,
    db_paths: Sequence[Path] | None = None,
    db_pairs: Sequence[tuple[Path, Path]] | None = None,
) -> StepResult:
    """Collect eval-role macro/micro best-parameter pairs.

    Args:
        config: Validated modelbridge configuration.
        db_paths: Optional explicit DB path list.
        db_pairs: Optional explicit macro/micro DB pair list.

    Returns:
        StepResult: Persisted step result for ``collect_eval``.
    """
    return _collect_role(config, role="eval", db_paths=tuple(db_paths or ()), db_pairs=tuple(db_pairs or ()))


def _collect_role(
    config: BridgeConfig,
    *,
    role: Role,
    db_paths: tuple[Path, ...],
    db_pairs: tuple[tuple[Path, Path], ...],
) -> StepResult:
    """Collect macro/micro pairs for one role across all scenarios."""
    output_dir = config.bridge.output_dir
    scenario_outputs: dict[str, dict[str, Any]] = {}
    issues: list[str] = []

    for scenario in config.bridge.scenarios:
        output = scenario_path(output_dir, scenario.name)
        output.mkdir(parents=True, exist_ok=True)
        pairs, source = _collect_pairs_for_scenario(
            output_dir=output_dir,
            scenario_name=scenario.name,
            role=role,
            scenario_output=output,
            db_paths=db_paths,
            db_pairs=db_pairs,
        )

        csv_path = output / ("train_pairs.csv" if role == "train" else "test_pairs.csv")
        if pairs:
            write_csv(csv_path, _pairs_to_rows(pairs))
            scenario_outputs[scenario.name] = {
                "status": "success",
                "pairs_csv": str(csv_path),
                "num_pairs": len(pairs),
                "source": source,
            }
            continue

        scenario_outputs[scenario.name] = {
            "status": "missing",
            "pairs_csv": str(csv_path),
            "num_pairs": 0,
            "source": source,
        }
        issues.append(f"{scenario.name}:{role} has no valid macro/micro pairs")

    return finalize_scenario_step(
        output_dir=output_dir,
        step=f"collect_{role}",
        strict_mode=config.bridge.strict_mode,
        scenario_outputs=scenario_outputs,
        issues=issues,
        inputs={
            "role": role,
            "db_paths": [str(path) for path in db_paths],
            "db_pairs": [[str(macro), str(micro)] for macro, micro in db_pairs],
        },
    )


def _collect_pairs_for_scenario(
    *,
    output_dir: Path,
    scenario_name: str,
    role: Role,
    scenario_output: Path,
    db_paths: tuple[Path, ...],
    db_pairs: tuple[tuple[Path, Path], ...],
) -> tuple[list[Pair], str]:
    """Collect pairs for one scenario using configured source priority."""
    if db_pairs:
        return _pairs_from_explicit_pairs(db_pairs), "explicit_db_pairs"
    if db_paths:
        return _pairs_from_paths(scenario_name, role, db_paths), "explicit_db_paths"

    plan_paths = _load_db_paths_from_plan(output_dir, scenario_name, role)
    if plan_paths:
        pairs = _pairs_from_paths(scenario_name, role, plan_paths)
        if pairs:
            return pairs, "plan"

    run_root = scenario_output / "runs" / role
    return _pairs_from_paths(scenario_name, role, list(run_root.rglob("optuna.db"))), "layout_scan"


def _load_db_paths_from_plan(output_dir: Path, scenario_name: str, role: Role) -> list[Path]:
    """Load existing DB paths for one scenario from role plan."""
    source_plan = plan_path(output_dir, role)
    if not source_plan.exists():
        return []
    plan_role, entries = read_plan(source_plan)
    if plan_role != role:
        return []

    db_paths: list[Path] = []
    for entry in entries:
        if entry["scenario"] != scenario_name:
            continue
        db_path = Path(cast(str, entry["expected_db_path"]))
        if db_path.exists():
            db_paths.append(db_path)
    return db_paths


def _pairs_from_explicit_pairs(db_pairs: Sequence[tuple[Path, Path]]) -> list[Pair]:
    """Build pair payloads from explicit macro/micro DB path tuples."""
    pairs: list[Pair] = []
    for run_id, (macro_db, micro_db) in enumerate(db_pairs):
        macro_best = _load_best_params(macro_db, None)
        micro_best = _load_best_params(micro_db, None)
        if macro_best and micro_best:
            pairs.append((run_id, macro_best, micro_best))
    return pairs


def _pairs_from_paths(scenario_name: str, role: Role, db_paths: Sequence[Path]) -> list[Pair]:
    """Resolve valid macro/micro pairs from DB path candidates."""
    results: list[Pair] = []
    loaded_run_ids: set[int] = set()

    for run_id, macro_db, micro_db in _pair_db_paths(db_paths):
        macro_best = _load_best_params(macro_db, f"{scenario_name}-{role}-macro-{run_id:03d}")
        micro_best = _load_best_params(micro_db, f"{scenario_name}-{role}-micro-{run_id:03d}")
        if macro_best and micro_best:
            results.append((run_id, macro_best, micro_best))
            loaded_run_ids.add(run_id)

    for run_id, macro_db, macro_study, micro_db, micro_study in _pair_db_paths_by_study_name(
        scenario_name=scenario_name,
        role=role,
        db_paths=db_paths,
    ):
        if run_id in loaded_run_ids:
            continue
        macro_best = _load_best_params(macro_db, macro_study)
        micro_best = _load_best_params(micro_db, micro_study)
        if macro_best and micro_best:
            results.append((run_id, macro_best, micro_best))

    return sorted(results, key=lambda item: item[0])


def _pair_db_paths(db_paths: Sequence[Path]) -> list[tuple[int, Path, Path]]:
    """Pair macro/micro DB paths using run directory layout."""
    grouped: dict[int, dict[Target, Path]] = {}
    for db_path in db_paths:
        target = db_path.parent.name
        try:
            run_id = int(db_path.parent.parent.name)
        except ValueError:
            continue
        if target in {"macro", "micro"}:
            grouped.setdefault(run_id, {})[cast(Target, target)] = db_path

    pairs: list[tuple[int, Path, Path]] = []
    for run_id, targets in grouped.items():
        macro_db = targets.get("macro")
        micro_db = targets.get("micro")
        if macro_db and micro_db:
            pairs.append((run_id, macro_db, micro_db))
    return sorted(pairs, key=lambda item: item[0])


def _pair_db_paths_by_study_name(
    *,
    scenario_name: str,
    role: Role,
    db_paths: Sequence[Path],
) -> list[tuple[int, Path, str, Path, str]]:
    """Pair macro/micro DB paths by matching canonical study names."""
    grouped: dict[int, dict[Target, tuple[Path, str]]] = {}
    pattern = re.compile(rf"^{re.escape(scenario_name)}-{role}-(?P<target>macro|micro)-(?P<run_id>\d{{3}})$")

    for db_path in db_paths:
        for study_name in _list_studies(db_path):
            match = pattern.fullmatch(study_name)
            if not match:
                continue
            run_id = int(match.group("run_id"))
            grouped.setdefault(run_id, {})[cast(Target, match.group("target"))] = (db_path, study_name)

    pairs: list[tuple[int, Path, str, Path, str]] = []
    for run_id, targets in grouped.items():
        macro = targets.get("macro")
        micro = targets.get("micro")
        if macro and micro:
            pairs.append((run_id, macro[0], macro[1], micro[0], micro[1]))
    return sorted(pairs, key=lambda item: item[0])


def _pairs_to_rows(pairs: Sequence[Pair]) -> list[dict[str, Any]]:
    """Convert pair tuples into CSV row dictionaries."""
    rows: list[dict[str, Any]] = []
    for run_id, macro, micro in pairs:
        row: dict[str, Any] = {"run_id": run_id}
        row.update({f"macro_{name}": value for name, value in macro.items()})
        row.update({f"micro_{name}": value for name, value in micro.items()})
        rows.append(row)
    return rows


def _load_best_params(db_path: Path, study_name: str | None) -> dict[str, float] | None:
    """Load best-trial parameter mapping from one Optuna study DB."""
    storage_uri = f"sqlite:///{db_path.resolve()}"
    try:
        if study_name is None:
            studies = optuna.get_all_study_summaries(storage=storage_uri)
            if len(studies) != 1:
                return None
            study_name = studies[0].study_name
        study = optuna.load_study(study_name=study_name, storage=storage_uri)
        return {name: float(value) for name, value in study.best_trial.params.items()} if study.best_trial else None
    except Exception:
        return None


def _list_studies(db_path: Path) -> list[str]:
    """List study names available in one Optuna DB file."""
    storage_uri = f"sqlite:///{db_path.resolve()}"
    try:
        return [summary.study_name for summary in optuna.get_all_study_summaries(storage=storage_uri)]
    except Exception:
        return []
