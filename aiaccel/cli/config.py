#!/usr/bin/env python3
"""Config utilities CLI for aiaccel."""

import argparse
import sys

from aiaccel.config import load_config, print_config


def main() -> None:
    """Main entry point for aiaccel-config."""
    parser = argparse.ArgumentParser(
        prog="aiaccel-config",
        description="Config utilities for aiaccel",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # print command
    print_parser = subparsers.add_parser("print", help="Print config file")
    print_parser.add_argument("config", type=str, help="Config file in YAML format")
    print_parser.add_argument("--line-length", type=int, default=80, help="Line length for output")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "print":
        config = load_config(args.config)
        print_config(config, line_length=args.line_length)


if __name__ == "__main__":
    main()