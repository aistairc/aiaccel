"""CLI entry points exposed by :mod:`aiaccel.hpo.apps`."""

from .modelbridge import main as modelbridge
from .optimize import main as optimize

__all__ = [
    "modelbridge",
    "optimize",
]
