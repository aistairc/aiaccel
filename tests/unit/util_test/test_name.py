import pytest
from numpy.random import RandomState

from aiaccel.util import generate_random_name


def test_generate_random_name():
    with pytest.raises(TypeError):
        generate_random_name()

    with pytest.raises(TypeError):
        generate_random_name(initial=10)

    rng = RandomState(0)

    with pytest.raises(ValueError):
        generate_random_name(rng, 0)

    with pytest.raises(TypeError):
        generate_random_name(1, rng)

    assert generate_random_name(rng, 1) == "S"
