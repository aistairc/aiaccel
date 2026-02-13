from numpy.typing import NDArray

import numpy as np

import pytest
from sklearn.metrics import r2_score

import aiaccel.hpo.modelbridge.analyze as analyze_module
from aiaccel.hpo.modelbridge.analyze import (
    _fit_regression,
    _predict_regression,
    evaluate_metrics,
    evaluate_metrics_from_predictions,
)
from aiaccel.hpo.modelbridge.config import RegressionConfig


def test_regression_fit_and_predict() -> None:
    features = [{"x": float(x)} for x in np.linspace(0, 1, 5)]
    targets = [{"y": float(2 * x + 1)} for x in np.linspace(0, 1, 5)]

    model = _fit_regression(features, targets, RegressionConfig(kind="linear"))

    pred = _predict_regression(model, [{"x": 0.5}])
    assert abs(pred[0]["y"] - (2 * 0.5 + 1)) < 1e-6

    metrics = evaluate_metrics(model, features, targets, metrics=["mae"])
    assert metrics["mae"] < 1e-6


def test_polynomial_regression() -> None:
    features = [{"x": float(x)} for x in np.linspace(-1, 1, 10)]
    targets = [{"y": float(x * x - x + 0.5)} for x in np.linspace(-1, 1, 10)]

    model = _fit_regression(features, targets, RegressionConfig(kind="polynomial", degree=2))
    metrics = evaluate_metrics(model, features, targets, metrics=["mae"])
    assert metrics["mae"] < 1e-3


def test_regression_round_trip() -> None:
    features = [{"x": float(x)} for x in np.linspace(-1, 1, 5)]
    targets = [{"y": float(3 * x - 2)} for x in np.linspace(-1, 1, 5)]

    model = _fit_regression(features, targets, RegressionConfig(kind="linear"))
    expected = _predict_regression(model, [{"x": 0.25}])[0]["y"]

    import json

    dumped = json.dumps(model)
    loaded = json.loads(dumped)

    restored_pred = _predict_regression(loaded, [{"x": 0.25}])[0]["y"]
    assert restored_pred == pytest.approx(expected)


def test_regression_custom_metrics() -> None:
    features = [{"x": float(x)} for x in np.linspace(0, 1, 4)]
    targets = [{"y": float(2 * x)} for x in np.linspace(0, 1, 4)]

    model = _fit_regression(features, targets, RegressionConfig(kind="linear"))
    metrics = evaluate_metrics(model, features, targets, metrics=("mae",))
    assert set(metrics.keys()) == {"mae"}
    assert metrics["mae"] < 1e-6


def test_regression_mismatched_features_raises() -> None:
    features = [{"x": 0.0}, {"y": 1.0}]
    targets = [{"y": 0.0}, {"y": 1.0}]

    with pytest.raises(KeyError):
        _fit_regression(features, targets, RegressionConfig())


def test_gpr_regression() -> None:
    pytest.importorskip("GPy")
    features = [{"x": float(x)} for x in np.linspace(0, 1, 5)]
    targets = [{"y": float(np.sin(x))} for x in np.linspace(0, 1, 5)]

    model = _fit_regression(features, targets, RegressionConfig(kind="gpr", kernel="RBF", noise=1e-5))
    assert model["kernel"] == "RBF"
    assert model["noise"] == pytest.approx(1e-5)

    pred = _predict_regression(model, [{"x": 0.5}])[0]
    assert isinstance(pred["y"], float)

    import json

    dumped = json.dumps(model)
    loaded = json.loads(dumped)

    pred2 = _predict_regression(loaded, [{"x": 0.5}])[0]
    assert pred2["y"] == pytest.approx(pred["y"], rel=1e-3)


def test_gpr_regression_kernel_and_noise_propagated(monkeypatch: pytest.MonkeyPatch) -> None:
    features = [{"x": float(x)} for x in np.linspace(0, 1, 5)]
    targets = [{"y": float(3 * x - 1)} for x in np.linspace(0, 1, 5)]
    calls: list[tuple[str, float | None]] = []

    class _DummyKernel:
        def __init__(self, name: str, input_dim: int):
            self.name = name
            self.input_dim = input_dim

    class _DummyKern:
        @staticmethod
        def rbf(input_dim: int) -> _DummyKernel:
            return _DummyKernel("RBF", input_dim)

        @staticmethod
        def matern32(input_dim: int) -> _DummyKernel:
            return _DummyKernel("MATERN32", input_dim)

        @staticmethod
        def matern52(input_dim: int) -> _DummyKernel:
            return _DummyKernel("MATERN52", input_dim)

    _DummyKern.RBF = staticmethod(_DummyKern.rbf)  # type: ignore[attr-defined]
    _DummyKern.Matern32 = staticmethod(_DummyKern.matern32)  # type: ignore[attr-defined]
    _DummyKern.Matern52 = staticmethod(_DummyKern.matern52)  # type: ignore[attr-defined]

    class _DummyModel:
        def __init__(self, kernel: _DummyKernel):
            self.kernel = kernel

        def optimize(self, messages: bool = False) -> None:
            _ = messages

    class _DummyModels:
        @staticmethod
        def gp_regression(
            x_data: NDArray[np.float64],
            y_col: NDArray[np.float64],
            kernel: _DummyKernel,
            **kwargs: float,
        ) -> _DummyModel:
            _ = x_data
            _ = y_col
            calls.append((kernel.name, kwargs.get("noise_var")))
            return _DummyModel(kernel)

    _DummyModels.GPRegression = staticmethod(_DummyModels.gp_regression)  # type: ignore[attr-defined]

    class _DummyGPy:
        kern = _DummyKern
        models = _DummyModels

    monkeypatch.setattr(analyze_module, "GPy", _DummyGPy)
    monkeypatch.setattr("aiaccel.hpo.modelbridge.analyze.pickle.dumps", lambda _models: b"dummy")

    model = _fit_regression(
        features,
        targets,
        RegressionConfig(kind="gpr", kernel="Matern52", noise=2.5e-4),
    )

    assert model["kernel"] == "MATERN52"
    assert model["noise"] == pytest.approx(2.5e-4)
    assert len(calls) == 1
    kernel_name, noise_value = calls[0]
    assert kernel_name == "MATERN52"
    assert noise_value == pytest.approx(2.5e-4)


def test_constant_target_r2_matches_sklearn_semantics() -> None:
    targets = [{"y": 1.0}, {"y": 1.0}, {"y": 1.0}, {"y": 1.0}]
    predictions = [{"y": 0.0}, {"y": 1.0}, {"y": 2.0}, {"y": 1.0}]

    metrics = evaluate_metrics_from_predictions(targets, predictions, metrics=["r2"])

    y_true = np.asarray([[1.0], [1.0], [1.0], [1.0]], dtype=float)
    y_pred = np.asarray([[0.0], [1.0], [2.0], [1.0]], dtype=float)
    expected = float(r2_score(y_true, y_pred))

    assert metrics["r2"] == pytest.approx(expected)
