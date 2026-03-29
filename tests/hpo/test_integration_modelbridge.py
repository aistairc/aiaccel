"""Integration tests for the modelbridge pipeline.

Runs the full 6-stage pipeline end-to-end with a minimal synthetic objective:
  Stage 1: prepare   -- generate per-run HPO configs
  Stage 2: hpo-train -- run Optuna studies for train runs (macro & micro)
  Stage 3: hpo-test  -- run Optuna studies for test runs  (macro & micro)
  Stage 4: collect   -- extract best-param pairs to CSV
  Stage 5: fit       -- train regression model
  Stage 6: evaluate  -- assess predictions and export metrics

These tests verify that the modelbridge module functions as a whole
when upstream changes are made to other parts of aiaccel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

if TYPE_CHECKING:
    from typing import Any


from aiaccel.hpo.modelbridge import collect, evaluate, fit_model, prepare

# ─── Shared Objective Script ─────────────────────────────────────────────────

_OBJECTIVE_SCRIPT = """\
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("out_filename", type=Path)
parser.add_argument("--x", type=float, default=0.0)
parser.add_argument("--y", type=float, default=0.0)
args = parser.parse_args()

score = (args.x - 0.6) ** 2 + (args.y + 0.3) ** 2
args.out_filename.parent.mkdir(parents=True, exist_ok=True)
with open(args.out_filename, "w") as f:
    json.dump(score, f)
"""

# ─── Minimal Config ──────────────────────────────────────────────────────────

_N_TRAIN = 2
_N_TEST = 1
_N_TRIALS = 2


def _make_config(objective_script: Path) -> dict[str, Any]:
    return {
        "n_train": _N_TRAIN,
        "n_test": _N_TEST,
        "objective_command": [sys.executable, str(objective_script)],
        "train_macro_trials": _N_TRIALS,
        "train_micro_trials": _N_TRIALS,
        "test_macro_trials": _N_TRIALS,
        "test_micro_trials": _N_TRIALS,
        "train_params": {
            "macro": {"x": {"low": -1.0, "high": 1.0}},
            "micro": {"y": {"low": -1.0, "high": 1.0}},
        },
        "test_params": {
            "macro": {"x": {"low": -1.0, "high": 1.0}},
            "micro": {"y": {"low": -1.0, "high": 1.0}},
        },
    }


# ─── Pipeline Fixture ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def pipeline_workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Run the full 6-stage modelbridge pipeline once for all tests in this module.

    Returns:
        Path to the completed workspace directory.
    """
    root = tmp_path_factory.mktemp("modelbridge_integration")
    workspace = root / "workspace"
    workspace.mkdir()

    objective_script = root / "objective.py"
    objective_script.write_text(_OBJECTIVE_SCRIPT, encoding="utf-8")

    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(_make_config(objective_script)), encoding="utf-8")

    # Stage 1: prepare
    prepare.run_prepare(config_path=config_path, workspace=workspace)

    # Stages 2 & 3: HPO runs (train macro/micro and test macro/micro)
    for run_config in sorted((workspace / "runs").rglob("config.yaml")):
        subprocess.run(
            [
                sys.executable,
                "-c",
                "from aiaccel.hpo.apps.optimize import main; main()",
                "--config",
                str(run_config),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )

    # Stage 4: collect
    collect.run_collect(workspace=workspace, phase="train")
    collect.run_collect(workspace=workspace, phase="test")

    # Stage 5: fit
    fit_model.run_fit_model(workspace=workspace)

    # Stage 6: evaluate
    evaluate.run_evaluate(workspace=workspace)

    return workspace


# ─── Stage 1: Prepare ────────────────────────────────────────────────────────


def test_prepare_creates_train_macro_configs(pipeline_workspace: Path) -> None:
    for run_id in range(_N_TRAIN):
        path = pipeline_workspace / "runs" / "train" / "macro" / f"{run_id:03d}" / "config.yaml"
        assert path.exists(), f"Missing train macro config: {path}"


def test_prepare_creates_train_micro_configs(pipeline_workspace: Path) -> None:
    for run_id in range(_N_TRAIN):
        path = pipeline_workspace / "runs" / "train" / "micro" / f"{run_id:03d}" / "config.yaml"
        assert path.exists(), f"Missing train micro config: {path}"


def test_prepare_creates_test_macro_and_micro_configs(pipeline_workspace: Path) -> None:
    for target in ("macro", "micro"):
        path = pipeline_workspace / "runs" / "test" / target / "000" / "config.yaml"
        assert path.exists(), f"Missing test {target} config: {path}"


def test_prepare_generated_config_has_correct_n_trials(pipeline_workspace: Path) -> None:
    config_path = pipeline_workspace / "runs" / "train" / "macro" / "000" / "config.yaml"
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert loaded["n_trials"] == _N_TRIALS


def test_prepare_generated_config_contains_param_space(pipeline_workspace: Path) -> None:
    config_path = pipeline_workspace / "runs" / "train" / "macro" / "000" / "config.yaml"
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert "params" in loaded
    assert "x" in loaded["params"]


# ─── Stages 2 & 3: HPO ───────────────────────────────────────────────────────


def test_hpo_train_creates_optuna_dbs(pipeline_workspace: Path) -> None:
    for run_id in range(_N_TRAIN):
        for target in ("macro", "micro"):
            db = pipeline_workspace / "runs" / "train" / target / f"{run_id:03d}" / "optuna.db"
            assert db.exists(), f"Missing Optuna DB: {db}"


def test_hpo_test_creates_optuna_dbs(pipeline_workspace: Path) -> None:
    for target in ("macro", "micro"):
        db = pipeline_workspace / "runs" / "test" / target / "000" / "optuna.db"
        assert db.exists(), f"Missing Optuna DB: {db}"


def test_hpo_train_dbs_contain_completed_trials(pipeline_workspace: Path) -> None:
    import optuna

    db = pipeline_workspace / "runs" / "train" / "macro" / "000" / "optuna.db"
    storage = f"sqlite:///{db.resolve()}"
    summaries = optuna.get_all_study_summaries(storage=storage)
    assert summaries, "No study found in train macro DB"
    study = optuna.load_study(study_name=summaries[0].study_name, storage=storage)
    assert len(study.trials) >= 1


# ─── Stage 4: Collect ────────────────────────────────────────────────────────


def test_collect_creates_train_pairs_csv(pipeline_workspace: Path) -> None:
    assert (pipeline_workspace / "pairs" / "train_pairs.csv").exists()


def test_collect_train_pairs_has_expected_columns(pipeline_workspace: Path) -> None:
    csv_path = pipeline_workspace / "pairs" / "train_pairs.csv"
    with csv_path.open(encoding="utf-8") as f:
        fieldnames = csv.DictReader(f).fieldnames or []
    assert "run_id" in fieldnames
    assert "macro_x" in fieldnames
    assert "micro_y" in fieldnames


def test_collect_train_pairs_has_n_train_rows(pipeline_workspace: Path) -> None:
    csv_path = pipeline_workspace / "pairs" / "train_pairs.csv"
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == _N_TRAIN


def test_collect_creates_test_pairs_csv(pipeline_workspace: Path) -> None:
    assert (pipeline_workspace / "pairs" / "test_pairs.csv").exists()


def test_collect_test_pairs_has_n_test_rows(pipeline_workspace: Path) -> None:
    csv_path = pipeline_workspace / "pairs" / "test_pairs.csv"
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == _N_TEST


def test_collect_pairs_values_are_within_param_bounds(pipeline_workspace: Path) -> None:
    csv_path = pipeline_workspace / "pairs" / "train_pairs.csv"
    with csv_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            assert -1.0 <= float(row["macro_x"]) <= 1.0
            assert -1.0 <= float(row["micro_y"]) <= 1.0


# ─── Stage 5: Fit ────────────────────────────────────────────────────────────


def test_fit_creates_model_pkl(pipeline_workspace: Path) -> None:
    assert (pipeline_workspace / "models" / "regression_model.pkl").exists()


def test_fit_creates_model_meta_json(pipeline_workspace: Path) -> None:
    assert (pipeline_workspace / "models" / "model_meta.json").exists()


def test_fit_model_meta_has_required_keys(pipeline_workspace: Path) -> None:
    meta = json.loads((pipeline_workspace / "models" / "model_meta.json").read_text(encoding="utf-8"))
    assert "macro_features" in meta
    assert "micro_targets" in meta
    assert "model_type" in meta


def test_fit_model_meta_reflects_param_names(pipeline_workspace: Path) -> None:
    meta = json.loads((pipeline_workspace / "models" / "model_meta.json").read_text(encoding="utf-8"))
    # collect prefixes keys with "macro_" / "micro_"
    assert list(meta["macro_features"]) == ["macro_x"]
    assert list(meta["micro_targets"]) == ["micro_y"]


def test_fit_model_is_loadable_and_predicts(pipeline_workspace: Path) -> None:
    import pickle

    model_path = pipeline_workspace / "models" / "regression_model.pkl"
    with model_path.open("rb") as f:
        model = pickle.load(f)  # noqa: S301 - trusted local artifact.
    predictions = model.predict([[0.5]])
    assert predictions.shape == (1, 1)


# ─── Stage 6: Evaluate ───────────────────────────────────────────────────────


def test_evaluate_creates_predictions_csv(pipeline_workspace: Path) -> None:
    assert (pipeline_workspace / "pairs" / "test_predictions.csv").exists()


def test_evaluate_predictions_has_expected_columns(pipeline_workspace: Path) -> None:
    pred_csv = pipeline_workspace / "pairs" / "test_predictions.csv"
    with pred_csv.open(encoding="utf-8") as f:
        fieldnames = csv.DictReader(f).fieldnames or []
    assert "run_id" in fieldnames
    # evaluate uses "micro_<key>" prefix from model_meta
    assert "true_micro_y" in fieldnames
    assert "pred_micro_y" in fieldnames


def test_evaluate_predictions_row_count_matches_n_test(pipeline_workspace: Path) -> None:
    pred_csv = pipeline_workspace / "pairs" / "test_predictions.csv"
    with pred_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == _N_TEST


def test_evaluate_creates_summary_json(pipeline_workspace: Path) -> None:
    assert (pipeline_workspace / "models" / "summary.json").exists()


def test_evaluate_summary_has_valid_metrics(pipeline_workspace: Path) -> None:
    summary = json.loads((pipeline_workspace / "models" / "summary.json").read_text(encoding="utf-8"))
    assert "metrics" in summary
    metrics = summary["metrics"]
    for key in ("mse", "mae", "r2"):
        assert key in metrics, f"Missing metric: {key}"
        assert isinstance(metrics[key], float)
    assert metrics["mse"] >= 0.0
    assert metrics["mae"] >= 0.0


def test_evaluate_summary_n_test_samples(pipeline_workspace: Path) -> None:
    summary = json.loads((pipeline_workspace / "models" / "summary.json").read_text(encoding="utf-8"))
    assert summary["n_test_samples"] == _N_TEST


# ─── CLI Adapter (modelbridge.py) ────────────────────────────────────────────


def test_cli_prepare_dispatches_and_creates_configs(tmp_path: Path) -> None:
    """Verify the CLI adapter (aiaccel-hpo modelbridge prepare) runs end-to-end."""
    from aiaccel.hpo.apps.modelbridge import main as cli_main

    objective_script = tmp_path / "objective.py"
    objective_script.write_text(_OBJECTIVE_SCRIPT, encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_make_config(objective_script)), encoding="utf-8")
    workspace = tmp_path / "workspace"

    with pytest.raises(SystemExit) as exc_info:
        cli_main(["prepare", "--config", str(config_path), "--workspace", str(workspace)])
    assert exc_info.value.code == 0
    assert (workspace / "runs" / "train" / "macro" / "000" / "config.yaml").exists()


def test_cli_collect_dispatches_and_writes_csv(tmp_path: Path) -> None:
    """Verify the CLI adapter (aiaccel-hpo modelbridge collect) runs end-to-end."""
    from aiaccel.hpo.apps.modelbridge import main as cli_main

    # Set up a workspace with pre-built Optuna DBs (use the Python API for prepare + HPO)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    objective_script = tmp_path / "objective.py"
    objective_script.write_text(_OBJECTIVE_SCRIPT, encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_make_config(objective_script)), encoding="utf-8")

    prepare.run_prepare(config_path=config_path, workspace=workspace)
    for run_config in sorted((workspace / "runs" / "train").rglob("config.yaml")):
        subprocess.run(
            [sys.executable, "-c", "from aiaccel.hpo.apps.optimize import main; main()", "--config", str(run_config)],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )

    with pytest.raises(SystemExit) as exc_info:
        cli_main(["collect", "--workspace", str(workspace), "--phase", "train"])
    assert exc_info.value.code == 0
    assert (workspace / "pairs" / "train_pairs.csv").exists()
