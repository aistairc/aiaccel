from argparse import ArgumentParser

from hydra.utils import instantiate

from aiaccel.config import load_config

parser = ArgumentParser()
parser.add_argument("config", type=str, help="Config file in YAML format")
args, unk_args = parser.parse_known_args()

config = load_config(args.config)

model = instantiate(config.model)

print(model)
