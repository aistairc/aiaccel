from pathlib import Path
import aiaccel
from aiaccel.optimizer.optimizer import OptimizerLoeader
from importlib.machinery import SourceFileLoader


def test_OptimizerLoeader_1():
    algorithm = "aiaccel.optimizer.grid"
    opt = OptimizerLoeader(algorithm)
    assert opt.get() == aiaccel.optimizer.grid.search.Optimizer

    algorithm = "aiaccel.optimizer.random"
    opt = OptimizerLoeader(algorithm)
    assert opt.get() == aiaccel.optimizer.random.search.Optimizer

    algorithm = "aiaccel.optimizer.sobol"
    opt = OptimizerLoeader(algorithm)
    assert opt.get() == aiaccel.optimizer.sobol.search.Optimizer
    
    algorithm = "aiaccel.optimizer.nelder_mead"
    opt = OptimizerLoeader(algorithm)
    assert opt.get() == aiaccel.optimizer.nelder_mead.search.Optimizer

    algorithm = "aiaccel.optimizer.tpe"
    opt = OptimizerLoeader(algorithm)
    assert opt.get() == aiaccel.optimizer.tpe.search.Optimizer


def test_OptimizerLoeader_2():
    algorithm = Path("./tests/unit/optimizer_test/sample")
    try:
        opt = OptimizerLoeader(algorithm)
        assert True
    except:
        assert False
