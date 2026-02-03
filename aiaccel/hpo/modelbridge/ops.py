"Operational functions for modelbridge phases."

from __future__ import annotations

from typing import Any, cast

import base64
from collections.abc import Sequence
import os
from pathlib import Path
import pickle
import shlex
import subprocess

from omegaconf import DictConfig, OmegaConf

import numpy as np

import optuna
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

try:
    import GPy  # type: ignore
except ImportError:
    GPy = None

from .config import (
    DataAssimilationConfig,
    HpoSettings,
    ParameterBounds,
    RegressionConfig,
    ScenarioConfig,
)
from .utils import get_logger, hash_file, write_csv, write_json

_logger = get_logger(__name__)

# Stride to ensure unique seeds across different phases and targets.
# Warning: If runs exceeds this value, seeds may collide.
RUN_ID_OFFSET_STRIDE = 100000


# --- Setup Phase -------------------------------------------------------------


def run_setup_train_step(
    settings: HpoSettings,
    scenario: ScenarioConfig,
    runs: int,
    seed_base: int,
    scenario_dir: Path,
) -> None:
    """Generate HPO configs and directories for the train phase.

    Args:
        settings (HpoSettings): HPO configuration.
        scenario (ScenarioConfig): Scenario configuration.
        runs (int): Number of runs.
        seed_base (int): Base seed for RNG.
        scenario_dir (Path): Scenario output directory.
    """
    _run_setup_phase_common(settings, scenario, "train", runs, seed_base, scenario_dir)


def run_setup_eval_step(
    settings: HpoSettings,
    scenario: ScenarioConfig,
    runs: int,
    seed_base: int,
    scenario_dir: Path,
) -> None:
    """Generate HPO configs and directories for the eval phase.

    Args:
        settings (HpoSettings): HPO configuration.
        scenario (ScenarioConfig): Scenario configuration.
        runs (int): Number of runs.
        seed_base (int): Base seed for RNG.
        scenario_dir (Path): Scenario output directory.
    """
    _run_setup_phase_common(settings, scenario, "eval", runs, seed_base, scenario_dir)


def _run_setup_phase_common(
    settings: HpoSettings,
    scenario: ScenarioConfig,
    role: str,
    runs: int,
    seed_base: int,
    scenario_dir: Path,
) -> None:
    if runs <= 0:
        return

    _logger.info(f"Starting Setup step: scenario={scenario.name}, role={role}, runs={runs}")

    targets = ["macro", "micro"]

    for run_idx in range(runs):
        for target in targets:
            run_dir = scenario_dir / "runs" / role / f"{run_idx:03d}" / target
            run_dir.mkdir(parents=True, exist_ok=True)

            # Calculate base offset for seed generation
            base_group_idx = 0
            if role == "eval":
                base_group_idx += 2
            if target == "micro":
                base_group_idx += 1

            offset = base_group_idx * RUN_ID_OFFSET_STRIDE
            current_seed = seed_base + offset + run_idx

            params = scenario.train_params if role == "train" else scenario.eval_params
            space = params.macro if target == "macro" else params.micro

            if role == "train":
                trials = scenario.train_macro_trials if target == "macro" else scenario.train_micro_trials
                command = scenario.train_objective.command
            else:
                trials = scenario.eval_macro_trials if target == "macro" else scenario.eval_micro_trials
                command = scenario.eval_objective.command

            _generate_hpo_config(
                settings=settings,
                scenario_name=scenario.name,
                role=role,
                target=target,
                run_idx=run_idx,
                space=space,
                trials=trials,
                seed=current_seed,
                command=command,
                output_dir=run_dir,
            )
            _logger.info(f"Setup completed: {run_dir}")


def _generate_hpo_config(
    settings: HpoSettings,
    scenario_name: str,
    role: str,
    target: str,
    run_idx: int,
    space: dict[str, ParameterBounds],
    trials: int,
    seed: int,
    command: list[str],
    output_dir: Path,
) -> None:
    """Generate and save config.yaml for a single HPO run.

    Args:
        settings (HpoSettings): HPO configuration.
        scenario_name (str): Name of the scenario.
        role (str): "train" or "eval".
        target (str): "macro" or "micro".
        run_idx (int): Run index.
        space (dict[str, ParameterBounds]): Parameter search space.
        trials (int): Number of trials.
        seed (int): Random seed.
        command (list[str]): Objective function command.
        output_dir (Path): Output directory for this HPO run.
    """
    study_name = f"{scenario_name}-{role}-{target}-{run_idx:03d}"
    db_path = output_dir / "optuna.db"
    storage_uri = f"sqlite:///{db_path.resolve()}"

    base_conf = cast(DictConfig, OmegaConf.load(settings.base_config))
    if not isinstance(base_conf, DictConfig):
        raise ValueError("base_config must be a DictConfig")

    overrides = settings.macro_overrides if target == "macro" else settings.micro_overrides
    if overrides:
        base_conf = cast(DictConfig, OmegaConf.merge(base_conf, OmegaConf.create(overrides)))

    OmegaConf.update(base_conf, "n_trials", trials)
    OmegaConf.update(base_conf, "working_directory", str(output_dir))
    OmegaConf.update(base_conf, "command", command)

    # Clean up optimize/goal if needed, but usually kept from base config or overrides.
    # We ensure study settings are correct.
    optimize_conf = cast(dict[str, Any], base_conf.get("optimize", {}))
    goal = optimize_conf.get("goal", "minimize")
    direction = "minimize" if goal == "minimize" else "maximize"

    params_def: dict[str, Any] = {"_target_": "aiaccel.hpo.optuna.hparams_manager.HparamsManager"}
    for name, bounds in space.items():
        params_def[name] = {
            "_target_": "aiaccel.hpo.optuna.hparams.Float",
            "low": bounds.low,
            "high": bounds.high,
            "log": bounds.log,
            "step": bounds.step,
        }
    OmegaConf.update(base_conf, "params", params_def)

    study_conf = {
        "_target_": "optuna.create_study",
        "study_name": study_name,
        "storage": storage_uri,
        "direction": direction,
        "load_if_exists": True,
        "sampler": {
            "_target_": "optuna.samplers.TPESampler",
            "seed": seed,
        },
    }
    OmegaConf.update(base_conf, "study", study_conf)

    config_path = output_dir / "config.yaml"
    OmegaConf.save(base_conf, config_path)


# --- Regression Phase --------------------------------------------------------


def run_regression_step(scenario: ScenarioConfig, scenario_dir: Path) -> None:
    """Train regression model from HPO results.

    Args:
        scenario (ScenarioConfig): Scenario configuration.
        scenario_dir (Path): Scenario output directory.
    """
    _logger.info(f"Starting Regression step: scenario={scenario.name}")

    train_dir = scenario_dir / "runs" / "train"
    if not train_dir.exists():
        _logger.warning("No training data found (directory does not exist). Skipping regression.")
        return

    samples: list[tuple[int, dict[str, float], dict[str, float]]] = []

    for run_path in sorted(train_dir.iterdir()):
        if not run_path.is_dir():
            continue
        try:
            run_idx = int(run_path.name)
        except ValueError:
            continue

        macro_db = run_path / "macro" / "optuna.db"
        micro_db = run_path / "micro" / "optuna.db"

        if not macro_db.exists() or not micro_db.exists():
            continue

        macro_best = _load_best_param(macro_db, f"{scenario.name}-train-macro-{run_idx:03d}")
        micro_best = _load_best_param(micro_db, f"{scenario.name}-train-micro-{run_idx:03d}")

        if macro_best and micro_best:
            samples.append((run_idx, macro_best, micro_best))

    if not samples:
        _logger.warning("No valid training pairs found for regression. HPO might not have finished.")
        return

    pairs_data = []
    for rid, mac, mic in samples:
        row: dict[str, Any] = {"run_id": rid}
        row |= {f"macro_{k}": v for k, v in mac.items()}
        row |= {f"micro_{k}": v for k, v in mic.items()}
        pairs_data.append(row)
    write_csv(scenario_dir / "train_pairs.csv", pairs_data)

    model = _fit_regression([s[1] for s in samples], [s[2] for s in samples], scenario.regression)

    models_dir = scenario_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    write_json(models_dir / "regression_model.json", model)

    metrics = _evaluate_metrics(model, [s[1] for s in samples], [s[2] for s in samples], scenario.metrics)
    metrics_dir = scenario_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    write_json(metrics_dir / "train_metrics.json", metrics)


def _load_best_param(db_path: Path, study_name: str) -> dict[str, float] | None:
    """Load the best parameters from an Optuna study DB.

    Args:
        db_path (Path): Path to the Optuna SQLite DB.
        study_name (str): Name of the study within the DB.

    Returns:
        dict[str, float] | None: Best parameters if found, otherwise None.
    """
    storage = f"sqlite:///{db_path.resolve()}"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        if study.best_trial:
            return {k: float(v) for k, v in study.best_trial.params.items()}
    except Exception:
        return None
    return None


# --- Evaluation Phase --------------------------------------------------------


def run_evaluate_model_step(scenario: ScenarioConfig, scenario_dir: Path) -> None:
    """Evaluate regression model on eval HPO results.

    Args:
        scenario (ScenarioConfig): Scenario configuration.
        scenario_dir (Path): Scenario output directory.
    """
    _logger.info(f"Starting Evaluate Model step: scenario={scenario.name}")

    model_path = scenario_dir / "models" / "regression_model.json"
    if not model_path.exists():
        _logger.warning("No regression model found. Skipping evaluation.")
        return

    import json

    with model_path.open("r") as f:
        model_dict = json.load(f)

    # Using "eval" directory per specification / legacy
    eval_dir = scenario_dir / "runs" / "eval"
    if not eval_dir.exists():
        _logger.warning("No eval data found. Skipping evaluation.")
        return

    samples: list[tuple[int, dict[str, float], dict[str, float]]] = []

    for run_path in sorted(eval_dir.iterdir()):
        if not run_path.is_dir():
            continue
        try:
            run_idx = int(run_path.name)
        except ValueError:
            continue

        macro_db = run_path / "macro" / "optuna.db"
        micro_db = run_path / "micro" / "optuna.db"

        if not macro_db.exists() or not micro_db.exists():
            continue

        macro_best = _load_best_param(macro_db, f"{scenario.name}-eval-macro-{run_idx:03d}")
        micro_best = _load_best_param(micro_db, f"{scenario.name}-eval-micro-{run_idx:03d}")

        if macro_best and micro_best:
            samples.append((run_idx, macro_best, micro_best))

    if not samples:
        _logger.warning("No valid eval pairs found.")
        return

    # Predict
    features = [s[1] for s in samples]
    targets = [s[2] for s in samples]

    preds = _predict_regression(model_dict, features)

    metrics = _evaluate_metrics_from_preds(targets, preds, scenario.metrics)

    write_json(scenario_dir / "metrics" / "eval_metrics.json", metrics)

    pred_rows = []
    for i, (rid, mac, mic) in enumerate(samples):
        row: dict[str, Any] = {"run_id": rid}
        row |= {f"macro_{k}": v for k, v in mac.items()}
        row |= {f"actual_{k}": v for k, v in mic.items()}
        row |= {f"pred_{k}": v for k, v in preds[i].items()}
        pred_rows.append(row)
    write_csv(scenario_dir / "test_predictions.csv", pred_rows)


# --- Summary Phase -----------------------------------------------------------


def run_summary_step(scenarios: list[ScenarioConfig], output_dir: Path) -> None:
    """Aggregate results from all scenarios.

    Args:
        scenarios (list[ScenarioConfig]): List of scenarios.
        output_dir (Path): Output directory.
    """
    summary_data = {}

    for sc in scenarios:
        s_dir = output_dir / sc.name

        train_pairs = 0
        train_pairs_file = s_dir / "train_pairs.csv"
        if train_pairs_file.exists():
            with train_pairs_file.open() as f:
                train_pairs = sum(1 for _ in f) - 1

        eval_pairs = 0
        test_preds_file = s_dir / "test_predictions.csv"
        if test_preds_file.exists():
            with test_preds_file.open() as f:
                eval_pairs = sum(1 for _ in f) - 1

        train_metrics = {}
        tm_file = s_dir / "metrics" / "train_metrics.json"
        if tm_file.exists():
            import json

            with tm_file.open() as f:
                train_metrics = json.load(f)

        eval_metrics = {}
        em_file = s_dir / "metrics" / "eval_metrics.json"
        if em_file.exists():
            import json

            with em_file.open() as f:
                eval_metrics = json.load(f)

        summary_data[sc.name] = {
            "train_pairs": max(0, train_pairs),
            "eval_pairs": max(0, eval_pairs),
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
        }

    write_json(output_dir / "summary.json", summary_data)


# --- Data Assimilation Phase -------------------------------------------------


def run_da_step(config: DataAssimilationConfig) -> None:
    """Execute external data assimilation command using aiaccel-job.

    Args:
        config (DataAssimilationConfig): Configuration.
    """
    if not config.enabled:
        return

    _logger.info(f"Starting Data Assimilation: {config.command}")

    if config.output_root is None:
        raise ValueError("DataAssimilationConfig.output_root must be set before running DA")
    output_root = config.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    cwd = config.cwd.resolve() if config.cwd else Path.cwd()

    try:
        # Insert log file and -- separator for aiaccel-job
        log_file = output_root / "aiaccel_job_da.log"
        full_cmd = list(config.job_runner_command) + [str(log_file.resolve()), "--"] + config.command

        _logger.info(f"Executing external command: {shlex.join(full_cmd)}")
        proc = subprocess.run(
            full_cmd, cwd=cwd, env=os.environ | config.env, capture_output=True, text=True, check=True
        )
        _logger.info(f"DA stdout:\n{proc.stdout}")
        if proc.stderr:
            _logger.warning(f"DA stderr:\n{proc.stderr}")

    except subprocess.CalledProcessError as exc:
        _logger.error(f"DA failed: {exc.stderr}")
        raise RuntimeError("Data assimilation command failed") from exc

    summary_path = output_root / "data_assimilation_summary.json"
    artifacts = []
    if summary_path.exists():
        artifacts.append(
            {"path": str(summary_path), "sha256": hash_file(summary_path), "size": summary_path.stat().st_size}
        )

    write_json(output_root / "data_assimilation_manifest.json", {"artifacts": artifacts})


# --- Regression Implementation Details ---------------------------------------


def _fit_regression(
    features_list: list[dict[str, float]],
    targets_list: list[dict[str, float]],
    config: RegressionConfig,
) -> dict[str, Any]:
    """Fit regression model and return serializable dict."""
    if not features_list:
        raise ValueError("No data to fit")

    feature_names = sorted(features_list[0].keys())
    target_names = sorted(targets_list[0].keys())

    x_data = np.asarray([[f[k] for k in feature_names] for f in features_list])
    y_data = np.asarray([[t[k] for k in target_names] for t in targets_list])

    kind = config.kind.lower()

    if kind in ["linear", "polynomial"]:
        degree = config.degree if kind == "polynomial" else 1
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        x_poly = poly.fit_transform(x_data)
        model = LinearRegression().fit(x_poly, y_data)

        feature_order = [
            label.replace(" ", "*").replace("^", "*") for label in poly.get_feature_names_out(feature_names)
        ]

        return {
            "kind": kind,
            "feature_names": feature_names,
            "target_names": target_names,
            "degree": degree,
            "feature_order": feature_order,
            "coefficients": model.coef_.tolist(),
            "intercept": model.intercept_.tolist(),
        }

    if kind == "gpr":
        if GPy is None:
            raise RuntimeError("GPy not installed")

        models = []
        for i in range(len(target_names)):
            y_col = y_data[:, i : i + 1]
            kernel = GPy.kern.RBF(input_dim=len(feature_names))
            m = GPy.models.GPRegression(x_data, y_col, kernel)
            m.optimize(messages=False)
            models.append(m)

        blob = base64.b64encode(pickle.dumps(models)).decode("utf-8")
        return {"kind": "gpr", "feature_names": feature_names, "target_names": target_names, "model_blob": blob}

    raise ValueError(f"Unknown regression kind: {kind}")


def _predict_regression(
    model_dict: dict[str, Any],
    features_list: list[dict[str, float]],
) -> list[dict[str, float]]:
    feature_names = model_dict["feature_names"]
    target_names = model_dict["target_names"]
    kind = model_dict["kind"]

    x_data = np.asarray([[f[k] for k in feature_names] for f in features_list])

    if kind in ["linear", "polynomial"]:
        degree = model_dict["degree"]
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        x_poly = poly.fit_transform(x_data)

        coef = np.asarray(model_dict["coefficients"])
        intercept = np.asarray(model_dict["intercept"])

        y_pred = x_poly @ coef.T + intercept

    elif kind == "gpr":
        if GPy is None:
            raise RuntimeError("GPy not installed")
        models = pickle.loads(base64.b64decode(model_dict["model_blob"]))
        preds = []
        for m in models:
            mean, _ = m.predict(x_data)
            preds.append(mean.flatten())
        y_pred = np.column_stack(preds)

    else:
        raise ValueError(f"Unknown model kind: {kind}")

    results = []
    for row in y_pred:
        results.append({k: float(v) for k, v in zip(target_names, row, strict=True)})
    return results


def _evaluate_metrics(
    model_dict: dict[str, Any],
    features_list: list[dict[str, float]],
    targets_list: list[dict[str, float]],
    metrics: Sequence[str],
) -> dict[str, float]:
    preds = _predict_regression(model_dict, features_list)
    return _evaluate_metrics_from_preds(targets_list, preds, metrics)


def _evaluate_metrics_from_preds(
    targets_list: list[dict[str, float]],
    preds_list: list[dict[str, float]],
    metrics: Sequence[str],
) -> dict[str, float]:
    target_names = sorted(targets_list[0].keys())

    y_true = np.asarray([[t[k] for k in target_names] for t in targets_list])
    y_pred = np.asarray([[p[k] for k in target_names] for p in preds_list])

    errors = y_true - y_pred
    result = {}

    if "mae" in metrics:
        result["mae"] = float(np.mean(np.abs(errors)))
    if "mse" in metrics:
        result["mse"] = float(np.mean(errors**2))
    if "r2" in metrics:
        var = np.var(y_true)
        if var == 0:
            result["r2"] = 1.0
        else:
            result["r2"] = 1.0 - float(np.mean(errors**2) / var)

    return result
