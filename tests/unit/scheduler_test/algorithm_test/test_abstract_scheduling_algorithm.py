from aiaccel.scheduler.algorithm.abstract_scheduling_algorithm import \
    AbstractSchedulingAlgorithm


def test_abstract_scheduling_algorithm(load_test_config):
    config = load_test_config()
    algorithm = AbstractSchedulingAlgorithm(config)

    try:
        algorithm.select_hp(None, None)
        assert False
    except NotImplementedError:
        assert True

    '''
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
    '''
