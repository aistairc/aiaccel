"""Collect best-parameter pairs from Optuna databases."""

from __future__ import annotations

from typing import Any

import argparse
from collections.abc import Sequence
import csv
from pathlib import Path

import optuna


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the collect step.

    Args:
        argv: Optional command-line arguments. When omitted, uses ``sys.argv``.

    Returns:
        Parsed command-line namespace.
    """
    parser = argparse.ArgumentParser(description="Modelbridge Collect Step")
    parser.add_argument("--workspace", type=str, required=True, help="Path to workspace directory")
    parser.add_argument("--phase", type=str, required=True, choices=["train", "test"], help="Phase (train or test)")
    return parser.parse_args(argv)


def extract_best_params(db_path: Path) -> dict[str, float] | None:
    """Extract best-trial parameters from one Optuna SQLite database.

    Args:
        db_path: Path to an ``optuna.db`` file.

    Returns:
        Best-trial parameter mapping when available. Returns ``None`` when the DB is missing,
        empty, unreadable, or contains non-float values.
    """
    if not db_path.exists():
        return None
    storage_uri = f"sqlite:///{db_path.resolve()}"
    try:
        studies = optuna.get_all_study_summaries(storage=storage_uri)
        if not studies:
            return None
        study = optuna.load_study(study_name=studies[0].study_name, storage=storage_uri)
        try:
            return {name: float(value) for name, value in study.best_trial.params.items()}
        except ValueError:
            return None
    except Exception as exc:  # pragma: no cover - depends on DB state corruption.
        print(f"Warning: Failed to read {db_path}: {exc}")
        return None


def run_collect(workspace: Path, phase: str) -> Path:
    """Collect macro/micro best-parameter pairs and write a CSV file.

    Args:
        workspace: Workspace directory containing ``runs/<phase>``.
        phase: Target phase name (``train`` or ``test``).

    Returns:
        Output CSV path under ``pairs/<phase>_pairs.csv``.

    Raises:
        FileNotFoundError: If ``runs/<phase>`` does not exist.
    """
    runs_dir = workspace / "runs" / phase
    out_csv = workspace / "pairs" / f"{phase}_pairs.csv"
    if not runs_dir.exists():
        raise FileNotFoundError(f"Directory not found: {runs_dir}")

    run_ids: set[str] = set()
    for db_path in runs_dir.rglob("optuna.db"):
        run_id = db_path.parent.name
        if run_id.isdigit():
            run_ids.add(run_id)

    records: list[dict[str, Any]] = []
    for run_id in sorted(run_ids):
        macro_db = runs_dir / "macro" / run_id / "optuna.db"
        micro_db = runs_dir / "micro" / run_id / "optuna.db"
        macro_params = extract_best_params(macro_db)
        micro_params = extract_best_params(micro_db)
        if macro_params is not None and micro_params is not None:
            row: dict[str, Any] = {"run_id": int(run_id)}
            row.update({f"macro_{key}": value for key, value in macro_params.items()})
            row.update({f"micro_{key}": value for key, value in micro_params.items()})
            records.append(row)
        else:
            print(f"Warning: Missing or incomplete best trial for run_id={run_id}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        print(f"Warning: No valid pairs found for phase={phase}")
        out_csv.touch()
        return out_csv

    fieldnames = {"run_id"}
    for record in records:
        fieldnames.update(record.keys())
    ordered_fieldnames = ["run_id", *sorted(name for name in fieldnames if name != "run_id")]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered_fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"[Collect] Extracted {len(records)} pairs to {out_csv}")
    return out_csv


def main(argv: Sequence[str] | None = None) -> int:
    """Run the collect step CLI.

    Args:
        argv: Optional command-line arguments. When omitted, uses ``sys.argv``.

    Returns:
        Process exit code. ``0`` on success, ``1`` on input/runtime errors.
    """
    args = parse_args(argv)
    try:
        run_collect(Path(args.workspace), args.phase)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
