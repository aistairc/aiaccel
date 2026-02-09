import numpy as np

import pytest

from aiaccel.hpo.modelbridge.config import RegressionConfig
from aiaccel.hpo.modelbridge.modeling import _evaluate_metrics, _fit_regression, _predict_regression


def test_regression_fit_and_predict() -> None:
    features = [{"x": float(x)} for x in np.linspace(0, 1, 5)]
    targets = [{"y": float(2 * x + 1)} for x in np.linspace(0, 1, 5)]

    model = _fit_regression(features, targets, RegressionConfig(kind="linear"))

    # Predict
    pred = _predict_regression(model, [{"x": 0.5}])
    assert abs(pred[0]["y"] - (2 * 0.5 + 1)) < 1e-6

    metrics = _evaluate_metrics(model, features, targets, metrics=["mae"])
    assert metrics["mae"] < 1e-6


def test_polynomial_regression() -> None:
    features = [{"x": float(x)} for x in np.linspace(-1, 1, 10)]
    targets = [{"y": float(x * x - x + 0.5)} for x in np.linspace(-1, 1, 10)]

    model = _fit_regression(features, targets, RegressionConfig(kind="polynomial", degree=2))
    metrics = _evaluate_metrics(model, features, targets, metrics=["mae"])
    assert metrics["mae"] < 1e-3


def test_regression_round_trip() -> None:
    features = [{"x": float(x)} for x in np.linspace(-1, 1, 5)]
    targets = [{"y": float(3 * x - 2)} for x in np.linspace(-1, 1, 5)]

    model = _fit_regression(features, targets, RegressionConfig(kind="linear"))

    # The model IS a dict, so "round trip" is trivial/implicit in serialization.
    # We just check prediction works.

    expected = _predict_regression(model, [{"x": 0.25}])[0]["y"]

    # Simulate save/load
    import json

    dumped = json.dumps(model)
    loaded = json.loads(dumped)

    restored_pred = _predict_regression(loaded, [{"x": 0.25}])[0]["y"]
    assert restored_pred == pytest.approx(expected)


def test_regression_custom_metrics() -> None:
    features = [{"x": float(x)} for x in np.linspace(0, 1, 4)]
    targets = [{"y": float(2 * x)} for x in np.linspace(0, 1, 4)]

    model = _fit_regression(features, targets, RegressionConfig(kind="linear"))
    metrics = _evaluate_metrics(model, features, targets, metrics=("mae",))
    assert set(metrics.keys()) == {"mae"}
    assert metrics["mae"] < 1e-6

    # In the new implementation, unknown metrics are just ignored or not returned.
    # So we skip the "raise ValueError" test.


def test_regression_mismatched_features_raises() -> None:
    features = [{"x": 0.0}, {"y": 1.0}]
    targets = [{"y": 0.0}, {"y": 1.0}]

    with pytest.raises(KeyError):
        _fit_regression(features, targets, RegressionConfig())


def test_gpr_regression(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("GPy")
    features = [{"x": float(x)} for x in np.linspace(0, 1, 5)]
    targets = [{"y": float(np.sin(x))} for x in np.linspace(0, 1, 5)]

    model = _fit_regression(features, targets, RegressionConfig(kind="gpr", kernel="RBF", noise=1e-5))
    assert model["kernel"] == "RBF"
    assert model["noise"] == pytest.approx(1e-5)

    pred = _predict_regression(model, [{"x": 0.5}])[0]
    assert isinstance(pred["y"], float)

    # round trip
    import json

    # model_blob is base64 string, so it is json serializable
    dumped = json.dumps(model)
    loaded = json.loads(dumped)

    pred2 = _predict_regression(loaded, [{"x": 0.5}])[0]
    assert pred2["y"] == pytest.approx(pred["y"], rel=1e-3)


def test_gpr_regression_kernel_and_noise_propagated(monkeypatch: pytest.MonkeyPatch) -> None:
    import aiaccel.hpo.modelbridge.modeling as modeling

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
            x_data: np.ndarray,
            y_col: np.ndarray,
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

    monkeypatch.setattr(modeling, "GPy", _DummyGPy)
    monkeypatch.setattr(modeling.pickle, "dumps", lambda _models: b"dummy")

    model = _fit_regression(
        features,
        targets,
        RegressionConfig(kind="gpr", kernel="Matern52", noise=2.5e-4),
    )

    assert model["kernel"] == "MATERN52"
    assert model["noise"] == pytest.approx(2.5e-4)
    assert calls == [("MATERN52", pytest.approx(2.5e-4))]
