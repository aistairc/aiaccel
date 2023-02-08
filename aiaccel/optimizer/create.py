from __future__ import annotations

from importlib import import_module


def create_optimizer(search_algorithm: str) -> type | None:
    return import_and_getattr(search_algorithm)


def import_and_getattr(name: str) -> type | None:
    """Imports the specified Optimizer class.

    Args:
        name (str): Optimizer class name, e.g. aiaccel.optimizer.NelderMeadOptimizer

    Returns:
        type | None: Subclass of aiaccel.optimizer.abstract_optimizer.AbstractOptimizer.
    """
    module_name, attr_name = name.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, attr_name)
