# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import importlib

__all__ = ["modelbridge"]


def __getattr__(name):
    if name == "modelbridge":
        return importlib.import_module(".modelbridge", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
