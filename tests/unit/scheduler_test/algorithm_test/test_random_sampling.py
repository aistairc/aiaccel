import numpy as np

from aiaccel.scheduler import RandomSamplingSchedulingAlgorithm


def test_random_sampling(load_test_config):
    config = load_test_config()
    algorithm = RandomSamplingSchedulingAlgorithm(config)
    rng = np.random.RandomState(0)
    hp = [1]
    assert algorithm.select_hp(hp, rng=rng) == [1]

    """
    resource = {'numAvailable': 1}
    assert algorithm.select_one_hp(resource, hp) == 1
    assert algorithm.select_one_hp(resource, []) is None

    assert algorithm.select_n_hp(resource, hp, 1) == [1]
    assert algorithm.select_n_hp(resource, [], 1) is None
    """
