#! /usr/bin/env python3

# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing import Protocol, cast

from argparse import ArgumentParser
from collections.abc import Sequence
import importlib
from pathlib import Path
import pkgutil
import sys


class _AppsPackage(Protocol):
    __name__: str
    __path__: Sequence[str]


def _discover_apps(package: _AppsPackage) -> list[str]:
    """Return runnable app names for an ``aiaccel.<target>.apps`` package."""
    apps = []
    for _, name, ispkg in pkgutil.iter_modules(package.__path__):
        if ispkg:
            module = importlib.import_module(f"{package.__name__}.{name}")
            if not callable(getattr(module, "main", None)):
                continue
        apps.append(name.replace("_", "-"))
    return apps


def main() -> None:
    target_module = Path(sys.argv[0]).stem.split("-")[-1]

    package = cast(_AppsPackage, importlib.import_module(f"aiaccel.{target_module}.apps"))

    modules = _discover_apps(package)
    if not modules:
        raise RuntimeError(f"No apps found in aiaccel.{target_module}.apps")

    parser = ArgumentParser(description=f"Run aiaccel-{target_module} apps.", add_help=False)
    parser.add_argument("command", choices=modules, help="The command to run.")
    args, unk_args = parser.parse_known_args()

    module = importlib.import_module(f"aiaccel.{target_module}.apps.{args.command.replace('-', '_')}")

    sys.argv = [str(module.__file__)] + unk_args
    module.main()


if __name__ == "__main__":
    main()
