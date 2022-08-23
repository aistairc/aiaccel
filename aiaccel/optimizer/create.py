from typing import Any
from aiaccel.config import Config
from importlib import import_module


def create_optimizer(config_path: str) -> Any:
    config = Config(config_path)
    return import_and_getattr(config.search_algorithm.get())


def import_and_getattr(name: str) -> Any:
    """ Imports the specified Optimizer class.
-
    Args:
        name(str): Optimizer class name
            (e.g.) aiaccel.optimizer.NelderMeadOptimizer
-
    Returns:
        Any: <Optimizer class>
    """
    module_name, attr_name = name.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, attr_name)
