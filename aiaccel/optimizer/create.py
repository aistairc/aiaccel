from __future__ import annotations

from typing import Any

from importlib import import_module

from aiaccel.config import Config


def create_optimizer(config_path: str) -> Any:
    """Returns master type.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        type | None: Subclass of aiaccel.optimizer.abstract_optimizer.AbstractOptimizer.
    """
    config = Config(config_path)
    return import_and_getattr(config.search_algorithm.get())


def import_and_getattr(name: str) -> Any:
    """Imports the specified Optimizer class.

    Args:
        name (str): Optimizer class name, e.g. aiaccel.optimizer.NelderMeadOptimizer

    Returns:
        type | None: Subclass of aiaccel.optimizer.abstract_optimizer.AbstractOptimizer.
    """
    module_name, attr_name = name.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, attr_name)
