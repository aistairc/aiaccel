# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing import Any


def modelbridge(*args: Any, **kwargs: Any) -> Any:
    from .modelbridge import main

    return main(*args, **kwargs)


def optimize(*args: Any, **kwargs: Any) -> Any:
    from .optimize import main

    return main(*args, **kwargs)


__all__ = [
    "modelbridge",
    "optimize",
]
