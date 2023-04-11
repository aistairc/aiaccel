import numpy as np
import pytest

from aiaccel.scheduler import AbstractSchedulingAlgorithm


def test_abstract_scheduling_algorithm(load_test_config):
    config = load_test_config()
    algorithm = AbstractSchedulingAlgorithm(config)
    rng = np.random.RandomState(0)

    with pytest.raises(NotImplementedError):
        algorithm.select_hp(None, None, rng=rng)

    """
    try:
        algorithm.select_one_hp(None, None)
        assert False
    except NotImplementedError:
        assert True

    try:
        algorithm.select_n_hp(None, None, None)
        assert False
    except NotImplementedError:
        assert True
    """
