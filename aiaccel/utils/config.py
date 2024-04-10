from __future__ import annotations

import re
from copy import deepcopy
from pathlib import Path

from colorama import Fore
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as oc


def print_config(config: ListConfig | DictConfig, line_length: int = 80) -> None:
    config = pathlib2str_config(config)  # https://github.com/omry/omegaconf/issues/82

    print("=" * line_length)
    for line in oc.to_yaml(config).splitlines():
        print(re.sub(r"(\s*)(\w+):", rf"\1{Fore.YELLOW}\2{Fore.RESET}:", line, count=1))
    print("=" * line_length)


def pathlib2str_config(config: ListConfig | DictConfig) -> ListConfig | DictConfig:
    def _inner_fn(config: ListConfig | DictConfig) -> ListConfig | DictConfig:
        if isinstance(config, ListConfig):
            for ii in range(len(config)):
                config[ii] = _inner_fn(config[ii])
        elif isinstance(config, DictConfig):
            for k, v in config.items():
                if isinstance(v, ListConfig | DictConfig):
                    config[k] = _inner_fn(v)
                elif isinstance(v, Path):
                    config[k] = str(v)

        return config

    return _inner_fn(deepcopy(config))
