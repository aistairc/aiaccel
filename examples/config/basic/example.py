# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from argparse import ArgumentParser

from hydra.utils import instantiate

from aiaccel.config import prepare_config, print_config

parser = ArgumentParser()
parser.add_argument("config", type=str, help="Config file in YAML format")
args, unk_args = parser.parse_known_args()

config = prepare_config(args.config)
print_config(config)

model = instantiate(config.model)

print(model)
