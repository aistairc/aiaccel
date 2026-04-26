# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

def modelbridge(*args, **kwargs):
    from .modelbridge import main

    return main(*args, **kwargs)


def optimize(*args, **kwargs):
    from .optimize import main

    return main(*args, **kwargs)


__all__ = [
    "modelbridge",
    "optimize",
]
