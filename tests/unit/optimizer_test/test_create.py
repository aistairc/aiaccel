from aiaccel.optimizer.nelder_mead_optimizer import NelderMeadOptimizer
from aiaccel.optimizer.random_optimizer import RandomOptimizer
from aiaccel.optimizer.sobol_optimizer import SobolOptimizer
from aiaccel.optimizer.grid_optimizer import GridOptimizer
from aiaccel.optimizer.tpe_optimizer import TpeOptimizer
from aiaccel.optimizer.create import create_optimizer

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
