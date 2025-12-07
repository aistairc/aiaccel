# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

import argparse

from omegaconf import OmegaConf as oc  # noqa: N813

from aiaccel.config.config import load_config, overwrite_omegaconf_dumper, resolve_inherit


def main() -> None:
    overwrite_omegaconf_dumper()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Configuration file path")
    parser.add_argument("key", help="Target key in the configration file")

    args, _ = parser.parse_known_args()
    config = load_config(args.config)
    config = resolve_inherit(config)

    print(oc.select(config, args.key))


if __name__ == "__main__":
    main()
