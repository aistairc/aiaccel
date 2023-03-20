from aiaccel.optimizer import NelderMeadOptimizer
from aiaccel.optimizer import RandomOptimizer
from aiaccel.optimizer import SobolOptimizer
from aiaccel.optimizer import GridOptimizer
from aiaccel.optimizer import TpeOptimizer
from aiaccel.optimizer import create_optimizer


def test_create():
    config = "tests/test_data/config_grid.json"
    assert create_optimizer(config) == GridOptimizer

    config = "tests/test_data/config_nelder-mead.json"
    assert create_optimizer(config) == NelderMeadOptimizer

    config = "tests/test_data/config_random.json"
    assert create_optimizer(config) == RandomOptimizer

    config = "tests/test_data/config_sobol.json"
    assert create_optimizer(config) == SobolOptimizer

    config = "tests/test_data/config_tpe.json"
    assert create_optimizer(config) == TpeOptimizer
