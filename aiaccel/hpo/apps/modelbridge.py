"""CLI adapter that routes to the modelbridge pipeline."""

from aiaccel.hpo.modelbridge.app import main as _main

__all__ = ["main"]


def main() -> None:
    """Entrypoint for ``aiaccel-hpo modelbridge``."""

    _main()
