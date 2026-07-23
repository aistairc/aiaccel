#! /usr/bin/env python3

# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser
import importlib
from pathlib import Path
import pkgutil
import sys


def main() -> None:
    target_module = Path(sys.argv[0]).stem.split("-")[-1]

    package = importlib.import_module(f"aiaccel.{target_module}.apps")

    modules = [name.replace("_", "-") for _, name, ispkg in pkgutil.iter_modules(package.__path__) if not ispkg]
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
