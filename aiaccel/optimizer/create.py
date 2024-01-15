from __future__ import annotations

from importlib import import_module
from typing import Type

from aiaccel.optimizer import AbstractOptimizer

# TODO: Replace typing.Type with builtins.type when aiaccel supports python>=3.9.
OptimizerType = Type[AbstractOptimizer]


def create_optimizer(search_algorithm: str) -> type:
    """Creates an optimizer class.

    Args:
        search_algorithm (str): Optimizer class name, e.g. aiaccel.optimizer.NelderMeadOptimizer

    Returns:
        type: Subclass of aiaccel.optimizer.abstract_optimizer.AbstractOptimizer.
    """
    return import_and_getattr(search_algorithm)


def import_and_getattr(name: str) -> OptimizerType:
    """Imports the specified Optimizer class.

    Args:
        name (str): Optimizer class name, e.g. aiaccel.optimizer.NelderMeadOptimizer

    Returns:
        type | None: Subclass of aiaccel.optimizer.abstract_optimizer.AbstractOptimizer.
    """
    module_name, attr_name = name.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, attr_name)
