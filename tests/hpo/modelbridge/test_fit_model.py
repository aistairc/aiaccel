from __future__ import annotations

import json
from pathlib import Path

import pytest

from aiaccel.hpo.modelbridge import fit_model


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


def test_run_fit_model_creates_model_and_meta(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    train_pairs = workspace / "pairs" / "train_pairs.csv"
    _write_train_pairs(train_pairs)

    model_path = fit_model.run_fit_model(workspace)
    meta_path = workspace / "models" / "model_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    assert model_path == workspace / "models" / "regression_model.pkl"
    assert model_path.exists()
    assert meta["macro_features"] == ["macro_lr"]
    assert meta["micro_targets"] == ["micro_lr", "micro_momentum"]
    assert meta["n_samples"] == 3


def test_run_fit_model_rejects_missing_train_pairs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Training data not found"):
        fit_model.run_fit_model(tmp_path / "workspace")


def test_run_fit_model_returns_none_for_empty_csv(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    csv_path = workspace / "pairs" / "train_pairs.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("run_id,macro_lr,micro_lr\n", encoding="utf-8")

    assert fit_model.run_fit_model(workspace) is None
