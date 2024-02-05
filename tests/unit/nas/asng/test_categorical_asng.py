import numpy as np
import pytest

from aiaccel.nas.asng.categorical_asng import CategoricalASNG


def test_categorical_asng():
    categories = np.array([2, 3, 4])
    params = np.zeros([len(categories), max(categories)])
    asng = CategoricalASNG(categories, params)

    assert np.sum(categories - 1) == asng.N
    assert asng.delta == 1.0
    assert asng.Delta == 1.0
    assert asng.delta_max == np.inf
    assert asng.alpha == 1.5
    assert asng.gamma == 0.0
    assert np.array_equal(asng.s, np.zeros(asng.N))
    assert np.array_equal(asng.params, np.zeros([len(categories), max(categories)]))
    assert np.array_equal(asng.params_max, np.array([[1e-10], [1e-10], [1e-10]]))

    assert asng.get_delta() == asng.delta / asng.Delta

    samples = asng.sampling()
    assert samples.shape == (len(categories), 4)

    ms = np.random.rand(10, len(categories), max(categories))
    losses = np.random.rand(10)
    asng.update(ms, losses)

    f = np.random.rand(10)
    w, idx = CategoricalASNG.utility(f)
    assert w.shape == (10,)
    assert idx.shape == (10,)
