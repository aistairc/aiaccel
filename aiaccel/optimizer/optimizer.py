from functools import singledispatchmethod
from importlib.machinery import SourceFileLoader
from importlib import import_module
from pathlib import Path
from typing import Union


class OptimizerLoeader:
    def __init__(self, algorithm: Union[str, Path]):
        self.opt = self.load(algorithm)

    def get(self):
        return self.opt

    @singledispatchmethod
    def load(self, algorithm: str):
        mod = algorithm + ".search"
        return import_module(mod)

    @load.register
    def _(self, algorithm: Path):
        path = algorithm / "search.py"
        return SourceFileLoader('optimizer', str(path)).load_module()
