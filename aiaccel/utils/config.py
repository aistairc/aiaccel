from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path

from colorama import Fore
from omegaconf import OmegaConf as oc
from omegaconf import ListConfig, DictConfig


def print_config(config: ListConfig | DictConfig, line_length: int = 80) -> None:
    config = pathlib2str_config(deepcopy(config))  # https://github.com/omry/omegaconf/issues/82

    print("=" * line_length)
    for line in oc.to_yaml(config).splitlines():
        print(re.sub(r"(\s*)(\w+):", rf"\1{Fore.YELLOW}\2{Fore.RESET}:", line, count=1))
    print("=" * line_length)


def pathlib2str_config(config: ListConfig | DictConfig):
    if isinstance(config, ListConfig):
        for k in range(len(config)):
            config[k] = pathlib2str_config(config[k])
    elif isinstance(config, DictConfig):
        for k, v in config.items():
            if isinstance(v, (ListConfig, DictConfig)):
                config[k] = pathlib2str_config(v)
            elif isinstance(v, Path):
                config[k] = str(v)
    
    return config