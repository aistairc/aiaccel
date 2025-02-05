from typing import Any

from copy import deepcopy
from pathlib import Path
import re

from colorama import Fore
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as oc  # noqa:N813


def load_config(
    config_filename: Path, bootstrap_config: dict[str, Any] | DictConfig | ListConfig | None = None
) -> DictConfig | ListConfig:
    """Load YAML configuration

    When the user specifies ``_base_``, the specified YAML file is loaded as the base,
    and the original configuration is merged with the base config.
    If the configuration specified in ``_base_`` also contains ``_base_``, the process is handled recursively.

    Additionally, if `bootstrap_config` is provided, it is merged with the final
    configuration to ensure any default values or overrides are applied.

    Args:
        config (Path): Path to the configuration
        bootstrap_config (dict[str, Any] | DictConfig | ListConfig | None):
            A configuration that is always merged to the loaded configuration.
            This is intended to define default config paths (e.g., working_directory) dynamically.

    Returns:
        merge_user_config (DictConfig): The merged configuration of the base config and the original config
        user_config(DictConfig | ListConfig) : The configuration without ``_base_``

    """

    if bootstrap_config is None:
        bootstrap_config = {}

    config = oc.merge(bootstrap_config, oc.load(config_filename))

    if isinstance(config, DictConfig):
        if "_base_" in config:
            base_path = Path(config["_base_"])

            config.pop("_base_")
            config = oc.merge(load_config(base_path, bootstrap_config), config)
        elif bootstrap_config is not None:
            config = oc.merge(bootstrap_config, config)

    return config


def print_config(config: ListConfig | DictConfig, line_length: int = 80) -> None:
    """
    Print the given configuration with syntax highlighting.

    This function converts `pathlib.Path` objects to strings before printing,
    ensuring that the output YAML format remains valid. It also highlights
    configuration keys in yellow for better readability.

    Args:
        config (ListConfig | DictConfig): The configuration to print.
        line_length (int, optional): The width of the separator line (default: 80).

    """

    config = pathlib2str_config(config)  # https://github.com/omry/omegaconf/issues/82

    print("=" * line_length)
    for line in oc.to_yaml(config).splitlines():
        print(re.sub(r"(\s*)(\w+):", rf"\1{Fore.YELLOW}\2{Fore.RESET}:", line, count=1))
    print("=" * line_length)


def pathlib2str_config(config: ListConfig | DictConfig) -> ListConfig | DictConfig:
    """
    Convert `pathlib.Path` objects in the configuration to strings.

    This function recursively traverses the configuration and replaces all `pathlib.Path`
    objects with their string representations. This is useful for saving the configuration
    in a YAML file, as YAML does not support `Path` objects.

    Args:
        config (ListConfig | DictConfig): The configuration to convert.

    Returns:
        ListConfig | DictConfig: The modified configuration with `Path` objects replaced by strings.

    """

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
