"""Regression strategies for modelbridge."""

from __future__ import annotations

from typing import Any

import base64
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import pickle

import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

try:  # optional dependency for Gaussian process regression
    import GPy  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    GPy = None

from .config import RegressionConfig
from .types import RegressionSample


@dataclass(slots=True)
class RegressionModel:
    """Serialized regression model supporting multiple backends."""

    kind: str
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    degree: int
    feature_order: tuple[tuple[str, ...], ...]
    coefficients: np.ndarray | None = None
    intercept: np.ndarray | None = None
    model_blob: str | None = None
    _gpr_models: list[Any] | None = field(default=None, repr=False, compare=False)

    def predict(self, features: Mapping[str, float]) -> dict[str, float]:
        if self.kind in {"linear", "polynomial"}:
            vector = _build_vector(features, self.feature_order)
            prediction = self.intercept + vector @ self.coefficients  # type: ignore[arg-type]
            return {name: float(value) for name, value in zip(self.target_names, prediction, strict=True)}
        if self.kind == "gpr":
            model_list = self._ensure_gpr_models()
            x = np.asarray([[float(features[name]) for name in self.feature_names]], dtype=float)
            preds = []
            for model in model_list:
                mean, _ = model.predict(x, include_likelihood=False)
                preds.append(float(mean.item()))
            return {name: value for name, value in zip(self.target_names, preds, strict=True)}
        raise ValueError(f"Unknown regression kind '{self.kind}'")

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "kind": self.kind,
            "feature_names": list(self.feature_names),
            "target_names": list(self.target_names),
            "degree": self.degree,
            "feature_order": ["*".join(names) for names in self.feature_order],
        }
        if self.coefficients is not None:
            payload["coefficients"] = self.coefficients.tolist()
        if self.intercept is not None:
            payload["intercept"] = self.intercept.tolist()
        if self.model_blob is not None:
            payload["model_blob"] = self.model_blob
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> RegressionModel:
        kind = str(payload["kind"])
        feature_names = tuple(str(name) for name in payload["feature_names"])
        target_names = tuple(str(name) for name in payload["target_names"])
        degree = int(payload.get("degree", 1))
        feature_order = tuple(
            tuple(filter(None, item.split("*")))
            for item in payload.get("feature_order", [])
        )
        coefficients = (
            np.asarray(payload["coefficients"], dtype=float)
            if "coefficients" in payload
            else None
        )
        intercept = (
            np.asarray(payload["intercept"], dtype=float)
            if "intercept" in payload
            else None
        )
        model_blob = payload.get("model_blob")
        model = cls(
            kind=kind,
            feature_names=feature_names,
            target_names=target_names,
            degree=degree,
            feature_order=feature_order,
            coefficients=coefficients,
            intercept=intercept,
            model_blob=model_blob,
        )
        return model

    def _ensure_gpr_models(self) -> list[Any]:
        if self._gpr_models is not None:
            return self._gpr_models
        if self.model_blob is None:
            raise ValueError("GPR model state is missing")
        if GPy is None:  # pragma: no cover - optional dependency
            raise RuntimeError("GPy is not installed. Install 'GPy' to use gpr regression.")
        data = base64.b64decode(self.model_blob.encode("utf-8"))
        self._gpr_models = pickle.loads(data)
        return self._gpr_models


def fit_regression(samples: Sequence[RegressionSample], config: RegressionConfig) -> RegressionModel:
    if not samples:
        raise ValueError("Regression requires at least one sample")

    feature_names = tuple(sorted(samples[0].features.keys()))
    if not feature_names:
        raise ValueError("Regression requires at least one feature")

    target_names = tuple(sorted(samples[0].target.keys()))
    if not target_names:
        raise ValueError("Regression target requires at least one parameter")

    reference_features = set(feature_names)
    reference_targets = set(target_names)
    for sample in samples:
        if set(sample.features.keys()) != reference_features:
            raise ValueError("Regression samples must share the same feature keys")
        if set(sample.target.keys()) != reference_targets:
            raise ValueError("Regression samples must share the same target keys")

    kind = config.kind.lower()
    if kind == "linear":
        return _fit_linear(samples, feature_names, target_names)
    if kind == "polynomial":
        return _fit_polynomial(samples, feature_names, target_names, degree=config.degree)
    if kind == "gpr":
        return _fit_gpr(samples, feature_names, target_names, config)

    raise ValueError(f"Unknown regression kind '{config.kind}'")


def evaluate_regression(
    model: RegressionModel,
    samples: Iterable[RegressionSample],
    *,
    metrics: Iterable[str] = ("mae", "mse", "r2"),
) -> dict[str, float]:
    y_true = []
    y_pred = []
    for sample in samples:
        y_true.append([float(sample.target[name]) for name in model.target_names])
        prediction = model.predict(sample.features)
        y_pred.append([prediction[name] for name in model.target_names])
    if not y_true:
        metric_names = list(metrics)
        return {metric: (0.0 if metric != "r2" else 1.0) for metric in metric_names}
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    errors = y_true_arr - y_pred_arr

    metric_funcs = {
        "mae": lambda: float(np.mean(np.abs(errors))),
        "mse": lambda: float(np.mean(np.square(errors))),
        "r2": lambda: 1.0
        if float(np.var(y_true_arr)) == 0
        else float(1.0 - float(np.mean(np.square(errors))) / float(np.var(y_true_arr))),
    }
    results: dict[str, float] = {}
    for name in metrics:
        if name not in metric_funcs:
            raise ValueError(f"Unsupported regression metric '{name}'")
        results[name] = metric_funcs[name]()
    return results


# --- Linear / Polynomial helpers -------------------------------------------------


def _fit_linear(
    samples: Sequence[RegressionSample],
    feature_names: tuple[str, ...],
    target_names: tuple[str, ...],
) -> RegressionModel:
    X = np.asarray([[float(sample.features[name]) for name in feature_names] for sample in samples])
    Y = np.asarray([[float(sample.target[name]) for name in target_names] for sample in samples])
    model = LinearRegression().fit(X, Y)
    feature_order = tuple((name,) for name in feature_names)
    return RegressionModel(
        kind="linear",
        feature_names=feature_names,
        target_names=target_names,
        degree=1,
        feature_order=feature_order,
        coefficients=np.asarray(model.coef_, dtype=float).T,
        intercept=np.asarray(model.intercept_, dtype=float),
    )


def _fit_polynomial(
    samples: Sequence[RegressionSample],
    feature_names: tuple[str, ...],
    target_names: tuple[str, ...],
    *,
    degree: int,
) -> RegressionModel:
    if degree < 1:
        raise ValueError("Polynomial regression requires degree >= 1")
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X = np.asarray([[float(sample.features[name]) for name in feature_names] for sample in samples])
    X_poly = poly.fit_transform(X)
    feature_labels = poly.get_feature_names_out(feature_names)
    feature_order = tuple(_parse_poly_feature(label) for label in feature_labels)
    Y = np.asarray([[float(sample.target[name]) for name in target_names] for sample in samples])
    lr = LinearRegression().fit(X_poly, Y)
    return RegressionModel(
        kind="polynomial",
        feature_names=feature_names,
        target_names=target_names,
        degree=degree,
        feature_order=feature_order,
        coefficients=np.asarray(lr.coef_, dtype=float).T,
        intercept=np.asarray(lr.intercept_, dtype=float),
    )


def _parse_poly_feature(label: str) -> tuple[str, ...]:
    components: list[str] = []
    for token in label.split(" "):
        token = token.strip()
        if not token:
            continue
        if "^" in token:
            name, exp = token.split("^")
            components.extend([name] * int(exp))
        else:
            components.append(token)
    return tuple(components)


# --- Gaussian process regression --------------------------------------------------


def _fit_gpr(
    samples: Sequence[RegressionSample],
    feature_names: tuple[str, ...],
    target_names: tuple[str, ...],
    config: RegressionConfig,
) -> RegressionModel:
    if GPy is None:  # pragma: no cover - optional dependency
        raise RuntimeError("GPy is not installed. Install 'GPy' to use gpr regression.")

    X = np.asarray([[float(sample.features[name]) for name in feature_names] for sample in samples], dtype=float)
    models: list[Any] = []
    for target in target_names:
        y = np.asarray([[float(sample.target[target])] for sample in samples], dtype=float)
        kernel = _build_kernel(feature_names, config)
        model = GPy.models.GPRegression(X, y, kernel)
        noise = config.noise if config.noise is not None else 1e-5
        model.Gaussian_noise.variance = noise
        model.optimize(messages=False)
        models.append(model)

    blob = base64.b64encode(pickle.dumps(models)).decode("utf-8")
    feature_order = tuple((name,) for name in feature_names)
    regression_model = RegressionModel(
        kind="gpr",
        feature_names=feature_names,
        target_names=target_names,
        degree=1,
        feature_order=feature_order,
        coefficients=None,
        intercept=None,
        model_blob=blob,
    )
    regression_model._gpr_models = models
    return regression_model


def _build_kernel(feature_names: Sequence[str], config: RegressionConfig) -> Any:  # pragma: no cover - requires GPy
    if GPy is None:
        raise RuntimeError("GPy is not installed")
    input_dim = len(feature_names)
    kernel_name = (config.kernel or "RBF").upper()
    if kernel_name == "RBF":
        return GPy.kern.RBF(input_dim=input_dim)
    if kernel_name == "MATERN32":
        return GPy.kern.Matern32(input_dim=input_dim)
    if kernel_name == "MATERN52":
        return GPy.kern.Matern52(input_dim=input_dim)
    raise ValueError(f"Unsupported GPR kernel '{config.kernel}'")


# --- utility ---------------------------------------------------------------------


def _build_vector(features: Mapping[str, float], order: Sequence[Sequence[str]]) -> np.ndarray:
    values = []
    for combo in order:
        value = 1.0
        for name in combo:
            value *= float(features[name])
        values.append(value)
    return np.asarray(values, dtype=float)


__all__ = ["RegressionModel", "fit_regression", "evaluate_regression"]
