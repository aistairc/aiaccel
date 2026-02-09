"""Regression modeling helpers for modelbridge."""

from __future__ import annotations

from typing import Any, Protocol

import base64
import pickle

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

try:
    import GPy  # type: ignore
except ImportError:
    GPy = None

from .config import RegressionConfig


class RegressorAdapter(Protocol):
    """Adapter interface for regression backends."""

    def fit(
        self,
        features: list[dict[str, float]],
        targets: list[dict[str, float]],
        config: RegressionConfig,
    ) -> dict[str, Any]:
        """Fit model and return serializable payload.

        Args:
            features: Feature dictionaries.
            targets: Target dictionaries.
            config: Regression configuration.

        Returns:
            dict[str, Any]: Serialized model payload.
        """

    def predict(self, model_payload: dict[str, Any], features: list[dict[str, float]]) -> list[dict[str, float]]:
        """Predict from model payload and feature dictionaries.

        Args:
            model_payload: Serialized model payload.
            features: Feature dictionaries.

        Returns:
            list[dict[str, float]]: Prediction dictionaries.
        """


class _BuiltinRegressorAdapter:
    """Adapter wrapping built-in regression implementations."""

    def __init__(self, kind: str):
        self._kind = kind

    def fit(
        self,
        features: list[dict[str, float]],
        targets: list[dict[str, float]],
        config: RegressionConfig,
    ) -> dict[str, Any]:
        normalized = config.model_copy(update={"kind": self._kind})
        return _fit_regression(features, targets, normalized)

    def predict(self, model_payload: dict[str, Any], features: list[dict[str, float]]) -> list[dict[str, float]]:
        return _predict_regression(model_payload, features)


def get_regressor_adapter(kind: str) -> RegressorAdapter:
    """Return built-in adapter for requested regression kind.

    Args:
        kind: Regression kind name.

    Returns:
        RegressorAdapter: Adapter implementation.

    Raises:
        ValueError: If kind is unsupported.
    """
    normalized = kind.strip().lower()
    if normalized not in {"linear", "polynomial", "gpr"}:
        raise ValueError(f"Unknown regression kind: {kind}")
    return _BuiltinRegressorAdapter(normalized)


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

        kernel_name = _normalize_gpr_kernel_name(config.kernel)
        noise_var = float(config.noise) if config.noise is not None else None

        models = []
        for i in range(len(target_names)):
            y_col = y_data[:, i : i + 1]
            kernel = _build_gpr_kernel(kernel_name, len(feature_names))
            gp_kwargs = {"noise_var": noise_var} if noise_var is not None else {}
            m = GPy.models.GPRegression(x_data, y_col, kernel, **gp_kwargs)
            m.optimize(messages=False)
            models.append(m)

        blob = base64.b64encode(pickle.dumps(models)).decode("utf-8")
        return {
            "kind": "gpr",
            "feature_names": feature_names,
            "target_names": target_names,
            "kernel": kernel_name,
            "noise": noise_var,
            "model_blob": blob,
        }

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
    metrics: list[str] | tuple[str, ...],
) -> dict[str, float]:
    preds = _predict_regression(model_dict, features_list)
    return _evaluate_metrics_from_preds(targets_list, preds, metrics)


def _evaluate_metrics_from_preds(
    targets_list: list[dict[str, float]],
    preds_list: list[dict[str, float]],
    metrics: list[str] | tuple[str, ...],
) -> dict[str, float]:
    target_names = sorted(targets_list[0].keys())

    y_true = np.asarray([[t[k] for k in target_names] for t in targets_list])
    y_pred = np.asarray([[p[k] for k in target_names] for p in preds_list])

    errors = y_true - y_pred
    result: dict[str, float] = {}

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


def _normalize_gpr_kernel_name(kernel: str | None) -> str:
    if kernel is None:
        return "RBF"

    normalized = kernel.strip().replace("_", "").replace("-", "").upper()
    if normalized in {"RBF", "MATERN32", "MATERN52"}:
        return normalized
    raise ValueError(f"Unsupported GPR kernel: {kernel}")


def _build_gpr_kernel(kernel_name: str, input_dim: int) -> Any:
    if GPy is None:
        raise RuntimeError("GPy not installed")

    if kernel_name == "RBF":
        return GPy.kern.RBF(input_dim=input_dim)
    if kernel_name == "MATERN32":
        return GPy.kern.Matern32(input_dim=input_dim)
    if kernel_name == "MATERN52":
        return GPy.kern.Matern52(input_dim=input_dim)
    raise ValueError(f"Unsupported GPR kernel: {kernel_name}")
