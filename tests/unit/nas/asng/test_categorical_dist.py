import numpy as np
import pytest

from aiaccel.nas.asng.categorical_dist import Categorical


def test_categorical():
    categories = [2, 3, 4]
    cat = Categorical(categories)

    assert cat.d == len(categories)
    assert categories == cat.C
    assert cat.Cmax == max(categories)
    assert cat.theta.shape == (cat.d, cat.Cmax)
    assert cat.valid_param_num == sum(c - 1 for c in categories)
    assert cat.valid_d == sum(1 for c in categories if c > 1)

    lam = 10
    samples = cat.sampling_lam(lam)
    assert samples.shape == (lam, cat.d, cat.Cmax)

    sample = cat.sampling()
    assert sample.shape == (cat.d, cat.Cmax)

    mle = cat.mle()
    assert mle.shape == (cat.d, cat.Cmax)

    X = np.random.rand(lam, cat.d, cat.Cmax)
    loglikelihoods = cat.loglikelihood(X)
    assert loglikelihoods.shape == (lam,)

    header = cat.log_header()
    assert len(header) == cat.d * len(cat.C)

    log = cat.log()
    assert len(log) == cat.d * len(cat.C)

    theta = np.zeros(cat.valid_param_num)
    cat.load_theta_from_log(theta)
    assert np.allclose(cat.theta, np.zeros((cat.d, cat.Cmax)))

    theta = np.zeros(cat.valid_param_num)
    cat._load_theta_from_log(theta)
    assert np.allclose(cat.theta, np.zeros((cat.d, cat.Cmax)))
