"""Analyze steps for regression fitting and evaluation."""

from __future__ import annotations

from typing import Any

import base64
from collections.abc import Sequence
import pickle

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures

try:
    import GPy  # type: ignore
except ImportError:
    GPy = None

from .common import StepResult, finalize_scenario_step, read_json, scenario_path, write_csv, write_json
from .config import BridgeConfig, RegressionConfig
from .pair_csv import parse_pairs_csv


def fit_regression(config: BridgeConfig) -> StepResult:
    """Fit per-scenario regression models from train pair CSV files."""
    output_dir = config.bridge.output_dir
    scenario_outputs: dict[str, dict[str, Any]] = {}
    issues: list[str] = []

    for scenario in config.bridge.scenarios:
        scenario_output_path = scenario_path(output_dir, scenario.name)
        train_pairs_path = scenario_output_path / "train_pairs.csv"
        if not train_pairs_path.exists():
            issues.append(f"{scenario.name}: missing train_pairs.csv")
            scenario_outputs[scenario.name] = {"status": "missing_pairs"}
            continue
        parsed = parse_pairs_csv(train_pairs_path)
        if not parsed:
            issues.append(f"{scenario.name}: train_pairs.csv has no valid rows")
            scenario_outputs[scenario.name] = {"status": "empty_pairs", "train_pairs_csv": str(train_pairs_path)}
            continue
        try:
            features, targets = [item[1] for item in parsed], [item[2] for item in parsed]
            model_payload = _fit_regression(features, targets, scenario.regression)
            metrics_payload = evaluate_metrics(model_payload, features, targets, list(scenario.metrics))
            model_path = write_json(scenario_output_path / "models" / "regression_model.json", model_payload)
            metrics_path = write_json(scenario_output_path / "metrics" / "train_metrics.json", metrics_payload)
            scenario_outputs[scenario.name] = {
                "status": "success",
                "model_path": str(model_path),
                "metrics_path": str(metrics_path),
            }
        except Exception as exc:
            error = {"stage": "fit_regression", "type": exc.__class__.__name__, "message": str(exc)}
            scenario_outputs[scenario.name] = {
                "status": "failed",
                "train_pairs_csv": str(train_pairs_path),
                "error": error,
            }
            issues.append(f"{scenario.name}: fit_regression failed ({error['type']}): {error['message']}")
    return finalize_scenario_step(
        output_dir=output_dir,
        step="fit_regression",
        strict_mode=config.bridge.strict_mode,
        scenario_outputs=scenario_outputs,
        issues=issues,
    )


def evaluate_model(config: BridgeConfig) -> StepResult:
    """Evaluate fitted models against test pair CSV files."""
    output_dir = config.bridge.output_dir
    scenario_outputs: dict[str, dict[str, Any]] = {}
    issues: list[str] = []

    for scenario in config.bridge.scenarios:
        scenario_output_path = scenario_path(output_dir, scenario.name)
        model_path = scenario_output_path / "models" / "regression_model.json"
        eval_pairs_path = scenario_output_path / "test_pairs.csv"
        if not model_path.exists():
            issues.append(f"{scenario.name}: missing regression_model.json")
            scenario_outputs[scenario.name] = {"status": "missing_model"}
            continue
        if not eval_pairs_path.exists():
            issues.append(f"{scenario.name}: missing test_pairs.csv")
            scenario_outputs[scenario.name] = {"status": "missing_eval_pairs"}
            continue
        parsed = parse_pairs_csv(eval_pairs_path)
        if not parsed:
            issues.append(f"{scenario.name}: test_pairs.csv has no valid rows")
            scenario_outputs[scenario.name] = {"status": "empty_eval_pairs"}
            continue
        try:
            model_payload = read_json(model_path)
            features, targets = [item[1] for item in parsed], [item[2] for item in parsed]
            predictions = _predict_regression(model_payload, features)
            metrics_payload = evaluate_metrics_from_predictions(targets, predictions, list(scenario.metrics))
            metrics_path = write_json(scenario_output_path / "metrics" / "eval_metrics.json", metrics_payload)
            rows: list[dict[str, Any]] = []
            for index, (run_id, macro, micro) in enumerate(parsed):
                row: dict[str, Any] = {"run_id": run_id}
                row.update({f"macro_{name}": value for name, value in macro.items()})
                row.update({f"actual_{name}": value for name, value in micro.items()})
                row.update({f"pred_{name}": value for name, value in predictions[index].items()})
                rows.append(row)
            predictions_path = write_csv(scenario_output_path / "test_predictions.csv", rows)
            scenario_outputs[scenario.name] = {
                "status": "success",
                "metrics_path": str(metrics_path),
                "predictions_path": str(predictions_path),
            }
        except Exception as exc:
            error = {"stage": "evaluate_model", "type": exc.__class__.__name__, "message": str(exc)}
            scenario_outputs[scenario.name] = {
                "status": "failed",
                "test_pairs_csv": str(eval_pairs_path),
                "error": error,
            }
            issues.append(f"{scenario.name}: evaluate_model failed ({error['type']}): {error['message']}")

    return finalize_scenario_step(
        output_dir=output_dir,
        step="evaluate_model",
        strict_mode=config.bridge.strict_mode,
        scenario_outputs=scenario_outputs,
        issues=issues,
    )


def _fit_regression(
    features_list: list[dict[str, float]],
    targets_list: list[dict[str, float]],
    config: RegressionConfig,
) -> dict[str, Any]:
    """Fit configured regression model and return serializable payload."""
    if not features_list:
        raise ValueError("No data to fit")

    feature_names = sorted(features_list[0].keys())
    target_names = sorted(targets_list[0].keys())
    x_data = np.asarray([[feature[name] for name in feature_names] for feature in features_list], dtype=float)
    y_data = np.asarray([[target[name] for name in target_names] for target in targets_list], dtype=float)
    kind = config.kind.lower()

    if kind in {"linear", "polynomial"}:
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
            "coefficients": np.asarray(model.coef_).tolist(),
            "intercept": np.asarray(model.intercept_).tolist(),
        }

    if kind == "gpr":
        if GPy is None:
            raise RuntimeError("GPy not installed")
        noise_var = float(config.noise) if config.noise is not None else None
        kernel_name, _ = _build_gpr_kernel(config.kernel, len(feature_names))
        models = []
        for index in range(len(target_names)):
            y_col = y_data[:, index : index + 1]
            _, kernel = _build_gpr_kernel(config.kernel, len(feature_names))
            model_kwargs = {"noise_var": noise_var} if noise_var is not None else {}
            gp_model = GPy.models.GPRegression(x_data, y_col, kernel, **model_kwargs)
            gp_model.optimize(messages=False)
            models.append(gp_model)

        model_blob = base64.b64encode(pickle.dumps(models)).decode("utf-8")
        return {
            "kind": "gpr",
            "feature_names": feature_names,
            "target_names": target_names,
            "kernel": kernel_name,
            "noise": noise_var,
            "model_blob": model_blob,
        }

    raise ValueError(f"Unknown regression kind: {config.kind}")


def _predict_regression(model_payload: dict[str, Any], features_list: list[dict[str, float]]) -> list[dict[str, float]]:
    """Predict target dictionaries from serialized model payload."""
    feature_names = list(model_payload["feature_names"])
    target_names = list(model_payload["target_names"])
    x_data = np.asarray([[feature[name] for name in feature_names] for feature in features_list], dtype=float)

    kind = str(model_payload["kind"])
    if kind in {"linear", "polynomial"}:
        poly = PolynomialFeatures(degree=int(model_payload["degree"]), include_bias=False)
        x_poly = poly.fit_transform(x_data)
        coefficients = np.asarray(model_payload["coefficients"], dtype=float)
        intercept = np.asarray(model_payload["intercept"], dtype=float)
        if coefficients.ndim == 1:
            coefficients = coefficients[np.newaxis, :]
        if intercept.ndim == 0:
            intercept = np.asarray([float(intercept)])
        y_pred = x_poly @ coefficients.T + intercept
    elif kind == "gpr":
        if GPy is None:
            raise RuntimeError("GPy not installed")
        models = pickle.loads(base64.b64decode(model_payload["model_blob"]))
        y_pred = np.column_stack([model.predict(x_data)[0].flatten() for model in models])
    else:
        raise ValueError(f"Unknown model kind: {kind}")

    return [{name: float(value) for name, value in zip(target_names, row, strict=True)} for row in y_pred]


def evaluate_metrics(
    model_payload: dict[str, Any],
    features: list[dict[str, float]],
    targets: list[dict[str, float]],
    metrics: Sequence[str],
) -> dict[str, float]:
    """Evaluate selected metrics from model predictions on features."""
    return evaluate_metrics_from_predictions(targets, _predict_regression(model_payload, features), metrics)


def evaluate_metrics_from_predictions(
    targets: list[dict[str, float]],
    predictions: list[dict[str, float]],
    metrics: Sequence[str],
) -> dict[str, float]:
    """Compute sklearn-compatible metrics from target/prediction pairs."""
    if not targets or not predictions:
        raise ValueError("targets and predictions must be non-empty")

    target_names = sorted(targets[0].keys())
    y_true = np.asarray([[row[name] for name in target_names] for row in targets], dtype=float)
    y_pred = np.asarray([[row[name] for name in target_names] for row in predictions], dtype=float)
    selected = set(metrics)
    metric_fns = {"mae": mean_absolute_error, "mse": mean_squared_error, "r2": r2_score}
    return {name: float(fn(y_true, y_pred)) for name, fn in metric_fns.items() if name in selected}


def _build_gpr_kernel(kernel: str | None, input_dim: int) -> tuple[str, Any]:
    """Create a GPy kernel from kernel name/alias."""
    if GPy is None:
        raise RuntimeError("GPy not installed")

    normalized = "RBF" if kernel is None else kernel.strip().replace("_", "").replace("-", "").upper()
    factories = {"RBF": GPy.kern.RBF, "MATERN32": GPy.kern.Matern32, "MATERN52": GPy.kern.Matern52}
    if normalized not in factories:
        raise ValueError(f"Unsupported GPR kernel: {kernel}")
    return normalized, factories[normalized](input_dim=input_dim)
