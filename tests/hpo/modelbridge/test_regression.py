import numpy as np

import pytest

from aiaccel.hpo.modelbridge.exceptions import ValidationError
from aiaccel.hpo.modelbridge.regression import evaluate, fit
from aiaccel.hpo.modelbridge.types import RegressionSample


def test_regression_fit_and_predict() -> None:
    samples = [RegressionSample(features={"x": float(x)}, target={"y": float(2 * x + 1)}) for x in np.linspace(0, 1, 5)]
    model = fit(samples)
    pred = model.predict({"x": 0.5})
    assert abs(pred["y"] - (2 * 0.5 + 1)) < 1e-6
    metrics = evaluate(model, samples)
    assert metrics["mae"] < 1e-6


def test_regression_mismatched_features_raises() -> None:
    samples = [
        RegressionSample(features={"x": 0.0}, target={"y": 0.0}),
        RegressionSample(features={"y": 1.0}, target={"y": 1.0}),
    ]
    with pytest.raises(ValidationError):
        fit(samples)


def test_regression_mismatched_targets_raises() -> None:
    samples = [
        RegressionSample(features={"x": 0.0}, target={"y": 0.0}),
        RegressionSample(features={"x": 1.0}, target={"z": 1.0}),
    ]
    with pytest.raises(ValidationError):
        fit(samples)
