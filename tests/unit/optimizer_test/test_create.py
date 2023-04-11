from aiaccel.optimizer import NelderMeadOptimizer
from aiaccel.optimizer import RandomOptimizer
from aiaccel.optimizer import SobolOptimizer
from aiaccel.optimizer import GridOptimizer
from aiaccel.optimizer import TpeOptimizer
from aiaccel.optimizer import create_optimizer


def test_create():
    assert create_optimizer('aiaccel.optimizer.GridOptimizer') == GridOptimizer
    assert create_optimizer('aiaccel.optimizer.NelderMeadOptimizer') == NelderMeadOptimizer
    assert create_optimizer('aiaccel.optimizer.RandomOptimizer') == RandomOptimizer
    assert create_optimizer('aiaccel.optimizer.SobolOptimizer') == SobolOptimizer
    assert create_optimizer('aiaccel.optimizer.TpeOptimizer') == TpeOptimizer
