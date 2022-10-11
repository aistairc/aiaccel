import random

from aiaccel.util.name import generate_random_name


def test_generate_random_name():
    assert generate_random_name(0) is None
    random.seed(1)
    assert generate_random_name(1) == 'i'
