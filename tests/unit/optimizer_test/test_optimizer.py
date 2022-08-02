from pathlib import Path
import aiaccel
from aiaccel.optimizer.optimizer import OptimizerLoeader


def test_OptimizerLoeader_1():
    algorithm = "aiaccel.optimizer.grid"
    opt = OptimizerLoeader(algorithm)
    assert opt.get().Optimizer == aiaccel.optimizer.grid.search.Optimizer

    algorithm = "aiaccel.optimizer.random"
    opt = OptimizerLoeader(algorithm)
    assert opt.get().Optimizer == aiaccel.optimizer.random.search.Optimizer

    algorithm = "aiaccel.optimizer.sobol"
    opt = OptimizerLoeader(algorithm)
    assert opt.get().Optimizer == aiaccel.optimizer.sobol.search.Optimizer
    
    algorithm = "aiaccel.optimizer.nelder_mead"
    opt = OptimizerLoeader(algorithm)
    assert opt.get().Optimizer == aiaccel.optimizer.nelder_mead.search.Optimizer
 
    algorithm = "aiaccel.optimizer.tpe"
    opt = OptimizerLoeader(algorithm)
    assert opt.get().Optimizer == aiaccel.optimizer.tpe.search.Optimizer
