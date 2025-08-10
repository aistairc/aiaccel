
from argparse import ArgumentParser

from aiaccel.config import (
    load_config,
    overwrite_omegaconf_dumper,
    print_config,
    resolve_inherit,
)
from hydra.utils import instantiate


overwrite_omegaconf_dumper()

parser = ArgumentParser()
parser.add_argument("config", type=str, help="Config file in YAML format")
args, unk_args = parser.parse_known_args()

config = load_config(args.config)
print_config(config)
config = resolve_inherit(config)

model = instantiate(config.model)

print(model)