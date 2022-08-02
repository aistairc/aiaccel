import copy
from pathlib import Path
from importlib.machinery import SourceFileLoader


class OptimizerLoeader:
    def __init__(self, optimizer_dir: Path):
        self.path = optimizer_dir / "search.py"

        if self.path.exists() is False:
            assert False

        self.opt = SourceFileLoader('optimizer', str(self.path)).load_module()

    def get(self):
        print(self.opt.__name__)
        return self.opt