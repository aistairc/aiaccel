import numpy as np

import pytest

from aiaccel.hpo.modelbridge.config import RegressionConfig
from aiaccel.hpo.modelbridge.regression import RegressionModel, evaluate_regression, fit_regression
from aiaccel.hpo.modelbridge.types import RegressionSample


def test_regression_fit_and_predict() -> None:
    samples = [RegressionSample(features={"x": float(x)}, target={"y": float(2 * x + 1)}) for x in np.linspace(0, 1, 5)]
    model = fit_regression(samples, RegressionConfig(kind="linear"))
    pred = model.predict({"x": 0.5})
    assert abs(pred["y"] - (2 * 0.5 + 1)) < 1e-6
    metrics = evaluate_regression(model, samples)
    assert metrics["mae"] < 1e-6


def test_polynomial_regression() -> None:
    samples = [
        RegressionSample(features={"x": float(x)}, target={"y": float(x * x - x + 0.5)})
        for x in np.linspace(-1, 1, 10)
    ]
    model = fit_regression(samples, RegressionConfig(kind="polynomial", degree=2))
    metrics = evaluate_regression(model, samples)
    assert metrics["mae"] < 1e-3


def test_regression_round_trip() -> None:
    samples = [RegressionSample(features={"x": float(x)}, target={"y": float(3 * x - 2)}) for x in np.linspace(-1, 1, 5)]
    model = fit_regression(samples, RegressionConfig(kind="linear"))
    restored = RegressionModel.from_dict(model.to_dict())
    expected = model.predict({"x": 0.25})["y"]
    assert restored.predict({"x": 0.25})["y"] == pytest.approx(expected)


def test_regression_custom_metrics() -> None:
    samples = [RegressionSample(features={"x": float(x)}, target={"y": float(2 * x)}) for x in np.linspace(0, 1, 4)]
    model = fit_regression(samples, RegressionConfig(kind="linear"))
    metrics = evaluate_regression(model, samples, metrics=("mae",))
    assert set(metrics.keys()) == {"mae"}
    assert metrics["mae"] < 1e-6
    with pytest.raises(ValueError):
        evaluate_regression(model, samples, metrics=("mae", "unknown"))


def test_regression_mismatched_features_raises() -> None:
    samples = [
        RegressionSample(features={"x": 0.0}, target={"y": 0.0}),
        RegressionSample(features={"y": 1.0}, target={"y": 1.0}),
    ]
    with pytest.raises(ValueError):
        fit_regression(samples, RegressionConfig())


def test_regression_mismatched_targets_raises() -> None:
    samples = [
        RegressionSample(features={"x": 0.0}, target={"y": 0.0}),
        RegressionSample(features={"x": 1.0}, target={"z": 1.0}),
    ]
    with pytest.raises(ValueError):
        fit_regression(samples, RegressionConfig())


def test_gpr_regression(monkeypatch) -> None:
    gpy = pytest.importorskip("GPy")
    samples = [
        RegressionSample(features={"x": float(x)}, target={"y": float(np.sin(x))})
        for x in np.linspace(0, 1, 5)
    ]
    model = fit_regression(samples, RegressionConfig(kind="gpr", kernel="RBF", noise=1e-5))
    pred = model.predict({"x": 0.5})
    assert isinstance(pred["y"], float)
    # ensure serialization round-trip works
    restored = RegressionModel.from_dict(model.to_dict())
    pred2 = restored.predict({"x": 0.5})
    assert pred2["y"] == pytest.approx(pred["y"], rel=1e-3)
