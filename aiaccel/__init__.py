# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version(__package__ or "aiaccel")
except PackageNotFoundError:
    __version__ = "0.0.0"
