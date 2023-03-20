import numpy as np

from aiaccel.scheduler.algorithm.abstract_scheduling_algorithm import AbstractSchedulingAlgorithm


def test_abstract_scheduling_algorithm(load_test_config):
    config = load_test_config()
    algorithm = AbstractSchedulingAlgorithm(config)
    rng = np.random.RandomState(0)

    try:
        algorithm.select_hp(None, None, rng=rng)
        assert False
    except NotImplementedError:
        assert True

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
