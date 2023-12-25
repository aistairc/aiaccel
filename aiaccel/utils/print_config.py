import re

from colorama import Fore
from omegaconf import OmegaConf as oc
from omegaconf import DictConfig


def print_config(config: DictConfig, line_length: int = 80):
    print("=" * line_length)
    for line in oc.to_yaml(config).splitlines():
        print(re.sub(r"(\s*)(\w+):", rf"\1{Fore.YELLOW}\2{Fore.RESET}:", line, count=1))
    print("=" * line_length)
