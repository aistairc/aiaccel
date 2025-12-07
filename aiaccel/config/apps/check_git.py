# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import argparse

from aiaccel.config.config import load_config
from aiaccel.config.git import collect_git_status_from_config, print_git_status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Configuration file path")

    args, _ = parser.parse_known_args()
    config = load_config(args.config)

    if len(git_status := collect_git_status_from_config(config)) > 0:
        print_git_status(git_status)

        exit(1)
    else:
        exit(0)


if __name__ == "__main__":
    main()
