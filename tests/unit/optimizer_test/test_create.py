from aiaccel.optimizer import (
    GridOptimizer,
    NelderMeadOptimizer,
    RandomOptimizer,
    SobolOptimizer,
    TpeOptimizer,
    create_optimizer,
)


def test_create():
    assert create_optimizer("aiaccel.optimizer.GridOptimizer") == GridOptimizer
    assert create_optimizer("aiaccel.optimizer.NelderMeadOptimizer") == NelderMeadOptimizer
    assert create_optimizer("aiaccel.optimizer.RandomOptimizer") == RandomOptimizer
    assert create_optimizer("aiaccel.optimizer.SobolOptimizer") == SobolOptimizer
    assert create_optimizer("aiaccel.optimizer.TpeOptimizer") == TpeOptimizer
