import numpy as np

from aiaccel.util.name import generate_random_name


def test_generate_random_name():
    rng = np.random.RandomState(0)
    assert generate_random_name(0, rng=rng) is None
    assert generate_random_name(1, rng=rng) == "S"
