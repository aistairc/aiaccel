from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from aiaccel.hpo.modelbridge import evaluate, fit_model


def _write_train_pairs(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "run_id,macro_lr,micro_lr,micro_momentum",
                "0,0.010,0.020,0.70",
                "1,0.020,0.030,0.75",
                "2,0.030,0.040,0.80",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_test_pairs(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "run_id,macro_lr,micro_lr,micro_momentum",
                "10,0.015,0.025,0.72",
                "11,0.025,0.035,0.78",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_run_evaluate_creates_predictions_and_summary(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _write_train_pairs(workspace / "pairs" / "train_pairs.csv")
    _write_test_pairs(workspace / "pairs" / "test_pairs.csv")
    fit_model.run_fit_model(workspace)

    summary_path = evaluate.run_evaluate(workspace)
    assert summary_path is not None
    prediction_path = workspace / "pairs" / "test_predictions.csv"
    rows = list(csv.DictReader(prediction_path.read_text(encoding="utf-8").splitlines()))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert summary_path == workspace / "models" / "summary.json"
    assert len(rows) == 2
    assert rows[0]["run_id"] == "10"
    assert "pred_micro_lr" in rows[0]
    assert "pred_micro_momentum" in rows[0]
    assert set(summary["metrics"].keys()) == {"mae", "mse", "r2"}
    assert summary["n_test_samples"] == 2


def test_run_evaluate_requires_model_artifacts(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _write_test_pairs(workspace / "pairs" / "test_pairs.csv")
    with pytest.raises(FileNotFoundError, match="Model or metadata"):
        evaluate.run_evaluate(workspace)


def test_run_evaluate_returns_none_for_empty_csv(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _write_train_pairs(workspace / "pairs" / "train_pairs.csv")
    fit_model.run_fit_model(workspace)
    test_path = workspace / "pairs" / "test_pairs.csv"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text("run_id,macro_lr,micro_lr,micro_momentum\n", encoding="utf-8")

    assert evaluate.run_evaluate(workspace) is None
