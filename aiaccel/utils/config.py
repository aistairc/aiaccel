from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re

from colorama import Fore
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as oc  # noqa:N813


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


def load_user_config(config: Path) -> DictConfig | ListConfig:
    """Load User Configuration

    When the user specifies _base_, the specified configuration is loaded as the base,
    and the original configuration is merged with base config.
    If the configuration specified in _base_ also contains _base_, the process is handled recursively.

    Args:
        config (Path): Path to the configuration

    Returns:
        merge_user_config (DictConfig): The merged configuration of the base config and the original config
        user_config(DictConfig | ListConfig) : The configuration without _base_

    """
    user_config = oc.load(config)
    if isinstance(user_config, DictConfig) and "_base_" in user_config:
        base_config = load_user_config(Path(user_config["_base_"]))
        merge_user_config = oc.merge(base_config, user_config)
        return merge_user_config
    else:
        return user_config
