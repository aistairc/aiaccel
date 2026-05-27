# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import argparse

from omegaconf import OmegaConf as oc  # noqa: N813

from aiaccel.config.config import prepare_config
from aiaccel.config.git import collect_git_status_from_config, print_git_status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Configuration file path")

    args, unk_args = parser.parse_known_args()
    config = prepare_config(args.config, overwrite_config=oc.from_cli(unk_args))

    git_status = collect_git_status_from_config(config)
    not_ready_status = [status for status in git_status if not status.ready()]

    if len(not_ready_status) > 0:
        print_git_status(git_status)

        exit(1)
    else:
        exit(0)


if __name__ == "__main__":
    main()
