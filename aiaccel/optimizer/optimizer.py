from functools import singledispatchmethod
from importlib.machinery import SourceFileLoader
from importlib import import_module
from pathlib import Path
from typing import Union
from typing import Any


class OptimizerLoeader:
    """
    The optimizer directory can be named arbitrarily.
    However, the file name of the program placed in the directory must be fixed to "search.py".

    The class name of the optimizer must be `Optimizer`.
    """
    def __init__(self, algorithm: Union[str, Path]) -> None:
        self.opt = self.load(algorithm)

    def get(self) -> Any:
        """Returns the loaded optimzier class.

        Args:
            None

        Returns:
            None
        """
        return self.opt.Optimizer

    @singledispatchmethod
    def load(self, algorithm: str) -> Any:
        """
        If config.optimizer.algorithm is set to use the built-in optimizer
        in aiaccel,this function is called.

        Args:
            None

        Returns:
            None
        """
        mod = algorithm + ".search"
        return import_module(mod)

    @load.register
    def _(self, algorithm: Path) -> Any:
        """
        This function is called if config.optimizer.algorithm is
        set to use an external optimizer.

        Args:
            None

        Returns:
            None
        """
        path = algorithm / "search.py"
        return SourceFileLoader('optimizer', str(path)).load_module()
