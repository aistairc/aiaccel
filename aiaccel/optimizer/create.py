from typing import Any
import aiaccel
from aiaccel.optimizer.grid.search import GridSearchOptimizer
from aiaccel.optimizer.nelder_mead.search import NelderMeadSearchOptimizer
from aiaccel.optimizer.random.search import RandomSearchOptimizer
from aiaccel.optimizer.sobol.search import SobolSearchOptimizer
from aiaccel.optimizer.tpe.search import TpeSearchOptimizer
from aiaccel.config import Config


def create_optimizer(config_path: str) -> Any:
    config = Config(config_path)
    algorithm = config.search_algorithm.get()

    # === grid search===
    if algorithm.lower() == aiaccel.search_algorithm_grid:
        return GridSearchOptimizer

    # === nelder-mead search===
    elif algorithm.lower() == aiaccel.search_algorithm_nelder_mead:
        return NelderMeadSearchOptimizer

    # === ramdom search===
    elif algorithm.lower() == aiaccel.search_algorithm_random:
        return RandomSearchOptimizer

    # === sobol search ===
    elif algorithm.lower() == aiaccel.search_algorithm_sobol:
        return SobolSearchOptimizer

    # === tpe search ===
    elif algorithm.lower() == aiaccel.search_algorithm_tpe:
        return TpeSearchOptimizer

    # === other (error) ===
    else:
        return None
