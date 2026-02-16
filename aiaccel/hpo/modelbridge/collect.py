"""Collect macro/micro best-parameter pairs from Optuna DB files."""

from __future__ import annotations

from typing import Any, cast

from collections.abc import Sequence
from pathlib import Path
import re

import optuna

from .common import Role, StepResult, Target, finalize_scenario_step, plan_path, read_plan, scenario_path, write_csv
from .config import BridgeConfig
from .pair_csv import PairRecord, pairs_to_rows

Diagnostic = dict[str, Any]


def collect_train(
    config: BridgeConfig,
    db_paths: Sequence[Path] | None = None,
    db_pairs: Sequence[tuple[Path, Path]] | None = None,
) -> StepResult:
    """Collect train-role macro/micro best-parameter pairs."""
    return _collect_role(config, role="train", db_paths=tuple(db_paths or ()), db_pairs=tuple(db_pairs or ()))


def collect_eval(
    config: BridgeConfig,
    db_paths: Sequence[Path] | None = None,
    db_pairs: Sequence[tuple[Path, Path]] | None = None,
) -> StepResult:
    """Collect eval-role macro/micro best-parameter pairs."""
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
        pairs, source, diagnostics = _collect_pairs_for_scenario(
            output_dir=output_dir,
            scenario_name=scenario.name,
            role=role,
            scenario_output=output,
            db_paths=db_paths,
            db_pairs=db_pairs,
        )

        csv_path = output / ("train_pairs.csv" if role == "train" else "test_pairs.csv")
        if pairs:
            write_csv(csv_path, pairs_to_rows(pairs))
            scenario_outputs[scenario.name] = {
                "status": "success",
                "pairs_csv": str(csv_path),
                "num_pairs": len(pairs),
                "source": source,
                "diagnostics": diagnostics,
            }
            continue

        scenario_outputs[scenario.name] = {
            "status": "missing",
            "pairs_csv": str(csv_path),
            "num_pairs": 0,
            "source": source,
            "diagnostics": diagnostics,
        }
        issues.append(
            _collect_issue_text(
                scenario_name=scenario.name,
                role=role,
                source=source,
                diagnostics=diagnostics,
            )
        )

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
) -> tuple[list[PairRecord], str, list[Diagnostic]]:
    """Collect pairs for one scenario using configured source priority."""
    if db_pairs:
        pairs, diagnostics = _pairs_from_explicit_pairs(db_pairs)
        return pairs, "explicit_db_pairs", diagnostics
    if db_paths:
        pairs, diagnostics = _pairs_from_paths(scenario_name, role, db_paths)
        return pairs, "explicit_db_paths", diagnostics

    plan_paths = _load_db_paths_from_plan(output_dir, scenario_name, role)
    fallback_diagnostics: list[Diagnostic] = []
    if plan_paths:
        pairs, diagnostics = _pairs_from_paths(scenario_name, role, tuple(plan_paths))
        if pairs:
            return pairs, "plan", diagnostics
        fallback_diagnostics.extend(diagnostics)

    run_root = scenario_output / "runs" / role
    pairs, diagnostics = _pairs_from_paths(scenario_name, role, tuple(sorted(run_root.rglob("optuna.db"))))
    return pairs, "layout_scan", [*fallback_diagnostics, *diagnostics]


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


def _pairs_from_explicit_pairs(db_pairs: Sequence[tuple[Path, Path]]) -> tuple[list[PairRecord], list[Diagnostic]]:
    """Build pair payloads from explicit macro/micro DB path tuples."""
    pairs: list[PairRecord] = []
    diagnostics: list[Diagnostic] = []
    for run_id, (macro_db, micro_db) in enumerate(db_pairs):
        macro_best, macro_reason = _load_best_params(macro_db, None)
        micro_best, micro_reason = _load_best_params(micro_db, None)
        if macro_reason is not None:
            diagnostics.append(
                {
                    "run_id": run_id,
                    "target": "macro",
                    "db_path": str(macro_db),
                    "reason": macro_reason,
                }
            )
        if micro_reason is not None:
            diagnostics.append(
                {
                    "run_id": run_id,
                    "target": "micro",
                    "db_path": str(micro_db),
                    "reason": micro_reason,
                }
            )
        if macro_best and micro_best:
            pairs.append((run_id, macro_best, micro_best))
    if not pairs and not diagnostics:
        diagnostics.append({"reason": "no_explicit_db_pairs"})
    return pairs, diagnostics


def _pairs_from_paths(
    scenario_name: str,
    role: Role,
    db_paths: Sequence[Path],
) -> tuple[list[PairRecord], list[Diagnostic]]:
    """Resolve valid macro/micro pairs from DB path candidates."""
    if not db_paths:
        return [], [{"reason": "no_db_candidates"}]

    results: list[PairRecord] = []
    diagnostics: list[Diagnostic] = []
    loaded_run_ids: set[int] = set()

    for run_id, macro_db, micro_db in _pair_db_paths(db_paths):
        macro_study = f"{scenario_name}-{role}-macro-{run_id:03d}"
        micro_study = f"{scenario_name}-{role}-micro-{run_id:03d}"
        pair, pair_diagnostics = _load_run_pair(
            run_id=run_id,
            macro_db=macro_db,
            macro_study=macro_study,
            micro_db=micro_db,
            micro_study=micro_study,
        )
        diagnostics.extend(pair_diagnostics)
        if pair is not None:
            results.append(pair)
            loaded_run_ids.add(run_id)

    study_pairs, study_diagnostics = _pair_db_paths_by_study_name(
        scenario_name=scenario_name,
        role=role,
        db_paths=db_paths,
    )
    diagnostics.extend(study_diagnostics)

    for run_id, macro_db, macro_study, micro_db, micro_study in study_pairs:
        if run_id in loaded_run_ids:
            continue
        pair, pair_diagnostics = _load_run_pair(
            run_id=run_id,
            macro_db=macro_db,
            macro_study=macro_study,
            micro_db=micro_db,
            micro_study=micro_study,
        )
        diagnostics.extend(pair_diagnostics)
        if pair is not None:
            results.append(pair)

    if not results and not diagnostics:
        diagnostics.append({"reason": "no_pairable_db_paths", "num_db_paths": len(db_paths)})
    return sorted(results, key=lambda item: item[0]), diagnostics


def _load_run_pair(
    *,
    run_id: int,
    macro_db: Path,
    macro_study: str,
    micro_db: Path,
    micro_study: str,
) -> tuple[PairRecord | None, list[Diagnostic]]:
    """Load one macro/micro pair with structured diagnostics."""
    diagnostics: list[Diagnostic] = []
    macro_best, macro_reason = _load_best_params(macro_db, macro_study)
    micro_best, micro_reason = _load_best_params(micro_db, micro_study)

    if macro_reason is not None:
        diagnostics.append(
            _load_diagnostic(
                run_id=run_id,
                target="macro",
                db_path=macro_db,
                study_name=macro_study,
                reason=macro_reason,
            )
        )
    if micro_reason is not None:
        diagnostics.append(
            _load_diagnostic(
                run_id=run_id,
                target="micro",
                db_path=micro_db,
                study_name=micro_study,
                reason=micro_reason,
            )
        )

    if macro_best is None or micro_best is None:
        return None, diagnostics
    return (run_id, macro_best, micro_best), diagnostics


def _load_diagnostic(
    *,
    run_id: int,
    target: Target,
    db_path: Path,
    study_name: str,
    reason: str,
) -> Diagnostic:
    """Build diagnostic mapping for one DB load attempt."""
    return {
        "run_id": run_id,
        "target": target,
        "db_path": str(db_path),
        "study_name": study_name,
        "reason": reason,
    }


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
) -> tuple[list[tuple[int, Path, str, Path, str]], list[Diagnostic]]:
    """Pair macro/micro DB paths by matching canonical study names."""
    grouped: dict[int, dict[Target, tuple[Path, str]]] = {}
    diagnostics: list[Diagnostic] = []
    pattern = re.compile(rf"^{re.escape(scenario_name)}-{role}-(?P<target>macro|micro)-(?P<run_id>\d{{3}})$")

    for db_path in db_paths:
        study_names, reason = _list_studies(db_path)
        if reason is not None:
            diagnostics.append({"db_path": str(db_path), "reason": reason})
        for study_name in study_names:
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
    return sorted(pairs, key=lambda item: item[0]), diagnostics


def _load_best_params(db_path: Path, study_name: str | None) -> tuple[dict[str, float] | None, str | None]:
    """Load best-trial parameters from one DB with failure reason."""
    if not db_path.exists():
        return None, "missing_db"

    storage_uri = f"sqlite:///{db_path.resolve()}"
    try:
        if study_name is None:
            studies = optuna.get_all_study_summaries(storage=storage_uri)
            if len(studies) != 1:
                return None, f"study_count_mismatch:{len(studies)}"
            study_name = studies[0].study_name
        study = optuna.load_study(study_name=study_name, storage=storage_uri)
        try:
            best_trial = study.best_trial
        except ValueError:
            return None, "best_trial_missing"
        return {name: float(value) for name, value in best_trial.params.items()}, None
    except Exception as exc:
        return None, f"{exc.__class__.__name__}:{exc}"


def _list_studies(db_path: Path) -> tuple[list[str], str | None]:
    """List study names available in one Optuna DB file."""
    if not db_path.exists():
        return [], "missing_db"

    storage_uri = f"sqlite:///{db_path.resolve()}"
    try:
        return [summary.study_name for summary in optuna.get_all_study_summaries(storage=storage_uri)], None
    except Exception as exc:
        return [], f"{exc.__class__.__name__}:{exc}"


def _collect_issue_text(scenario_name: str, role: Role, source: str, diagnostics: Sequence[Diagnostic]) -> str:
    """Build strict-mode issue message for missing pair outputs."""
    summary = f"collect_{role} scenario={scenario_name} role={role} source={source} has no valid macro/micro pairs"
    if not diagnostics:
        return summary
    preview = "; ".join(_format_diagnostic(diagnostic) for diagnostic in diagnostics[:3])
    if len(diagnostics) > 3:
        preview = f"{preview}; ... (+{len(diagnostics) - 3} more)"
    return f"{summary}: {preview}"


def _format_diagnostic(diagnostic: Diagnostic) -> str:
    """Render one diagnostic mapping into a compact text line."""
    parts: list[str] = []
    for key in ("db_path", "study_name", "run_id", "target", "reason", "num_db_paths"):
        if key in diagnostic:
            parts.append(f"{key}={diagnostic[key]}")
    if not parts:
        parts.append(str(diagnostic))
    return ", ".join(parts)
