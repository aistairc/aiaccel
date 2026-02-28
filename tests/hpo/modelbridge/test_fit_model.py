from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

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


def _write_regression_config(path: Path, *, kind: str, degree: int | None = None, noise: float | None = None) -> None:
    regression: dict[str, object] = {"kind": kind}
    if degree is not None:
        regression["degree"] = degree
    if noise is not None:
        regression["noise"] = noise
    config: dict[str, object] = {"regression": regression}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


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
    assert meta["regression_kind"] == "linear"
    assert meta["model_type"] == "LinearRegression"


def test_run_fit_model_rejects_missing_train_pairs(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Training data not found"):
        fit_model.run_fit_model(tmp_path / "workspace")


def test_run_fit_model_returns_none_for_empty_csv(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    csv_path = workspace / "pairs" / "train_pairs.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("run_id,macro_lr,micro_lr\n", encoding="utf-8")

    assert fit_model.run_fit_model(workspace) is None


def test_run_fit_model_supports_polynomial_regression(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _write_train_pairs(workspace / "pairs" / "train_pairs.csv")
    config_path = tmp_path / "config" / "config.yaml"
    _write_regression_config(config_path, kind="polynomial", degree=3)

    model_path = fit_model.run_fit_model(workspace, config_path=config_path)
    meta = json.loads((workspace / "models" / "model_meta.json").read_text(encoding="utf-8"))

    assert model_path == workspace / "models" / "regression_model.pkl"
    assert meta["regression_kind"] == "polynomial"
    assert meta["model_type"] == "PolynomialRegression"
    assert meta["regression_degree"] == 3


def test_run_fit_model_supports_gpr_regression(tmp_path: Path) -> None:
    pytest.importorskip("GPy")
    workspace = tmp_path / "workspace"
    _write_train_pairs(workspace / "pairs" / "train_pairs.csv")
    config_path = tmp_path / "config" / "config.yaml"
    _write_regression_config(config_path, kind="gpr", noise=1.0e-4)

    model_path = fit_model.run_fit_model(workspace, config_path=config_path)
    meta = json.loads((workspace / "models" / "model_meta.json").read_text(encoding="utf-8"))

    assert model_path == workspace / "models" / "regression_model.pkl"
    assert meta["regression_kind"] == "gpr"
    assert meta["model_type"] == "GPyGaussianProcessRegression"
    assert meta["gpr_noise"] == pytest.approx(1.0e-4)


def test_run_fit_model_rejects_unsupported_regression_kind(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _write_train_pairs(workspace / "pairs" / "train_pairs.csv")
    config_path = tmp_path / "config" / "config.yaml"
    _write_regression_config(config_path, kind="unsupported")

    with pytest.raises(ValueError, match="Unsupported regression kind"):
        fit_model.run_fit_model(workspace, config_path=config_path)


def test_run_fit_model_reads_regression_kind_from_config_file_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    _write_train_pairs(workspace / "pairs" / "train_pairs.csv")
    config_path = tmp_path / "config" / "config.yaml"
    _write_regression_config(config_path, kind="polynomial", degree=4)
    monkeypatch.setenv("CONFIG_FILE", str(config_path))

    fit_model.run_fit_model(workspace)
    meta = json.loads((workspace / "models" / "model_meta.json").read_text(encoding="utf-8"))

    assert meta["regression_kind"] == "polynomial"
    assert meta["regression_degree"] == 4
