from __future__ import annotations

import csv
from pathlib import Path

import optuna
import pytest

from aiaccel.hpo.modelbridge import collect


def _create_optuna_db(db_path: Path, *, study_name: str, param_names: tuple[str, ...]) -> None:
    storage_uri = f"sqlite:///{db_path.resolve()}"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_uri,
        direction="minimize",
        load_if_exists=True,
    )

    def objective(trial: optuna.trial.Trial) -> float:
        for name in param_names:
            trial.suggest_float(name, 0.0, 1.0)
        return 0.123

    study.optimize(objective, n_trials=1)


def test_run_collect_writes_pairs_csv(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _create_optuna_db(
        workspace / "runs" / "train" / "macro" / "000" / "optuna.db",
        study_name="train-macro-000",
        param_names=("lr",),
    )
    _create_optuna_db(
        workspace / "runs" / "train" / "micro" / "000" / "optuna.db",
        study_name="train-micro-000",
        param_names=("lr", "momentum"),
    )

    csv_path = collect.run_collect(workspace=workspace, phase="train")
    rows = list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))

    assert csv_path == workspace / "pairs" / "train_pairs.csv"
    assert len(rows) == 1
    assert rows[0]["run_id"] == "0"
    assert "macro_lr" in rows[0]
    assert "micro_lr" in rows[0]
    assert "micro_momentum" in rows[0]


def test_run_collect_touches_empty_csv_when_pairs_missing(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _create_optuna_db(
        workspace / "runs" / "test" / "macro" / "001" / "optuna.db",
        study_name="test-macro-001",
        param_names=("lr",),
    )

    csv_path = collect.run_collect(workspace=workspace, phase="test")
    assert csv_path.exists()
    assert csv_path.read_text(encoding="utf-8") == ""


def test_run_collect_requires_runs_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Directory not found"):
        collect.run_collect(workspace=tmp_path / "workspace", phase="train")
