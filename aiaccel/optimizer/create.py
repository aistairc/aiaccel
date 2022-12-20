from importlib import import_module
from typing import Any


def create_optimizer(search_algorithm: str) -> Any:
    return import_and_getattr(search_algorithm)


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
