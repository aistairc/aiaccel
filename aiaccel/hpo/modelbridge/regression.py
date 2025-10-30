"""Lightweight regression utilities backed by NumPy."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import itertools

import numpy as np

from .exceptions import ValidationError
from .types import RegressionSample


@dataclass(slots=True)
class RegressionModel:
    """Multi-output linear regression model."""

    coefficients: np.ndarray  # shape: (n_features, n_targets)
    intercept: np.ndarray  # shape: (n_targets,)
    feature_order: tuple[tuple[str, ...], ...]
    target_names: tuple[str, ...]
    degree: int

    def predict(self, features: Mapping[str, float]) -> dict[str, float]:
        """Predict micro parameters for the given macro ``features``."""

        vector = _build_vector(features, self.feature_order)
        prediction = self.intercept + vector @ self.coefficients
        return {name: float(value) for name, value in zip(self.target_names, prediction, strict=True)}

    def to_dict(self) -> dict[str, object]:
        """Serialise the model for persistence."""

        return {
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept.tolist(),
            "feature_order": ["*".join(names) for names in self.feature_order],
            "target_names": list(self.target_names),
            "degree": self.degree,
        }


def build_features(feature_names: Sequence[str], degree: int) -> tuple[tuple[str, ...], ...]:
    """Return ordered feature combinations for polynomial expansion."""

    order: list[tuple[str, ...]] = []
    for power in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(sorted(feature_names), power):
            order.append(combo)
    return tuple(order)


def fit(samples: Sequence[RegressionSample], degree: int = 1) -> RegressionModel:
    """Fit a regression model of ``degree`` using least squares."""

    if not samples:
        raise ValidationError("Regression requires at least one sample")
    feature_names = list(samples[0].features.keys())
    if not feature_names:
        raise ValidationError("Regression requires at least one feature")

    target_names = tuple(sorted(samples[0].target.keys()))
    if not target_names:
        raise ValidationError("Regression target requires at least one parameter")

    order = build_features(feature_names, degree)
    matrix = []
    targets: list[list[float]] = []
    for sample in samples:
        if set(sample.features.keys()) != set(feature_names):
            raise ValidationError("Regression samples must share the same feature keys")
        if set(sample.target.keys()) != set(target_names):
            raise ValidationError("Regression samples must share the same target keys")
        vector = _build_vector(sample.features, order)
        matrix.append(vector)
        targets.append([float(sample.target[name]) for name in target_names])
    mat_x = np.vstack(matrix)
    vec_y = np.asarray(targets, dtype=float)

    aug_x = np.hstack([np.ones((mat_x.shape[0], 1)), mat_x])
    coeffs, *_ = np.linalg.lstsq(aug_x, vec_y, rcond=None)
    intercept = coeffs[0, :]
    weights = coeffs[1:, :]
    return RegressionModel(
        coefficients=weights,
        intercept=intercept,
        feature_order=order,
        target_names=target_names,
        degree=degree,
    )


def evaluate(model: RegressionModel, samples: Iterable[RegressionSample]) -> dict[str, float]:
    """Compute MAE/MSE/R2 metrics for ``model`` against ``samples``."""

    y_true = []
    y_pred = []
    for sample in samples:
        y_true.append([float(sample.target[name]) for name in model.target_names])
        prediction = model.predict(sample.features)
        y_pred.append([prediction[name] for name in model.target_names])
    if not y_true:
        return {"mae": 0.0, "mse": 0.0, "r2": 1.0}
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true_arr - y_pred_arr)))
    mse = float(np.mean(np.square(y_true_arr - y_pred_arr)))
    denom = float(np.var(y_true_arr))
    r2 = 1.0 if denom == 0 else float(1.0 - mse / denom)
    return {"mae": mae, "mse": mse, "r2": r2}


def _build_vector(features: Mapping[str, float], order: Sequence[Sequence[str]]) -> np.ndarray:
    """Return a feature vector matching ``order`` for polynomial regression."""

    values = []
    for combo in order:
        value = 1.0
        for name in combo:
            value *= float(features[name])
        values.append(value)
    return np.asarray(values, dtype=float)


__all__ = ["RegressionModel", "build_features", "fit", "evaluate"]
