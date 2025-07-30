#! /usr/bin/env python3

from argparse import ArgumentParser
import importlib
from pathlib import Path
import pkgutil
import sys


def main() -> None:
    target_module = Path(sys.argv[0]).stem.split("-")[-1]

    package = importlib.import_module(f"aiaccel.{target_module}.apps")

    modules = [name for _, name, ispkg in pkgutil.iter_modules(package.__path__) if not ispkg]
    if not modules:
        raise RuntimeError(f"No apps found in aiaccel.{target_module}.apps")

    parser = ArgumentParser(description=f"Run aiaccel-{target_module} apps.", add_help=False)
    parser.add_argument("command", choices=modules, help="The command to run.")
    args, unk_args = parser.parse_known_args()

    sys.argv = [f"{sys.argv[0]}"] + [f"{args.command}"] + unk_args
    importlib.import_module(f"aiaccel.{target_module}.apps.{args.command}").main()


if __name__ == "__main__":
    main()
