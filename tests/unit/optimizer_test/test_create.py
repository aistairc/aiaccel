from aiaccel.optimizer.nelder_mead_optimizer import NelderMeadOptimizer
from aiaccel.optimizer.random_optimizer import RandomOptimizer
from aiaccel.optimizer.sobol_optimizer import SobolOptimizer
from aiaccel.optimizer.grid_optimizer import GridOptimizer
from aiaccel.optimizer.tpe_optimizer import TpeOptimizer
from aiaccel.optimizer.create import create_optimizer

def test_create():
    assert create_optimizer('aiaccel.optimizer.GridOptimizer') == GridOptimizer
    assert create_optimizer('aiaccel.optimizer.NelderMeadOptimizer') == NelderMeadOptimizer
    assert create_optimizer('aiaccel.optimizer.RandomOptimizer') == RandomOptimizer
    assert create_optimizer('aiaccel.optimizer.SobolOptimizer') == SobolOptimizer
    assert create_optimizer('aiaccel.optimizer.TpeOptimizer') == TpeOptimizer
