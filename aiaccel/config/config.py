from typing import Any, Callable

import copy
from copy import deepcopy
from pathlib import Path
import re

from colorama import Fore
from omegaconf import DictConfig, ListConfig
from omegaconf import Node
from omegaconf import OmegaConf as oc  # noqa:N813
from omegaconf._utils import OmegaConfDumper

import yaml
from yaml.resolver import BaseResolver


def overwrite_omegaconf_dumper(mode: str = "|") -> None:
    """
    Overwrites the default string representation in OmegaConf's YAML dumper.

    This function modifies the `OmegaConfDumper` to represent multi-line strings
    using the specified style (`mode`). By default, it uses the `|` block style
    for multi-line strings. Single-line strings remain unchanged.

    Args:
        mode (str, optional): The YAML style character for multi-line strings.
                              Defaults to "|".
    """

    def str_representer(dumper: OmegaConfDumper, data: str) -> yaml.Node:
        return dumper.represent_scalar(
            BaseResolver.DEFAULT_SCALAR_TAG, data, style=mode if len(data.splitlines()) > 1 else None
        )

    OmegaConfDumper.add_representer(str, str_representer)
    OmegaConfDumper.str_representer_added = True

def load_config(
    config_filename: str | Path,
    parent_config: dict[str, Any] | DictConfig | ListConfig | None = None,
) -> DictConfig | ListConfig:
    """Load YAML configuration

    When the user specifies ``_base_``, the specified YAML file is loaded as the base,
    and the original configuration is merged with the base config.
    If the configuration specified in ``_base_`` also contains ``_base_``, the process is handled recursively.

    Additionally, if `bootstrap_config` is provided, it is merged with the final
    configuration to ensure any default values or overrides are applied.

    Args:
        config (Path): Path to the configuration
        parent_config (dict[str, Any] | DictConfig | ListConfig | None):
            A configuration that is merged to the loaded configuration.
            This is intended to define default config paths (e.g., working_directory) dynamically.

    Returns:
        merge_user_config (DictConfig): The merged configuration of the base config and the original config
        user_config(DictConfig | ListConfig) : The configuration without ``_base_``

    """

    if not isinstance(config_filename, Path):
        config_filename = Path(config_filename)

    if not config_filename.is_absolute():
        config_filename = Path.cwd() / config_filename

    if parent_config is None:
        parent_config = {}

    config = oc.merge(oc.load(config_filename), parent_config)

    if isinstance(config, DictConfig) and "_base_" in config:
        # process _base_
        base_paths = config["_base_"]
        if not isinstance(base_paths, ListConfig):
            base_paths = [base_paths]

        config.pop("_base_")
        for base_path in map(Path, base_paths):
            if not base_path.is_absolute():
                base_path = config_filename.parent / base_path

            config = load_config(base_path, config)

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



def apply_recursively(func: Callable) -> DictConfig | ListConfig:
    """
    Recursively apply a function to all elements in a DictConfig or ListConfig.

    This function traverses the configuration structure and applies the provided
    function to each element, allowing for custom transformations or operations.

    Args:
        config (DictConfig | ListConfig): The configuration to traverse.
        func (callable): The function to apply to each element.

    Returns:
        DictConfig | ListConfig: The modified configuration with the function applied.
    """

    def _inner_fn(config: DictConfig | ListConfig) -> DictConfig | ListConfig:
        config = func(config)
        
        if isinstance(config, DictConfig):
            config = copy.deepcopy(config)

            for key in config:
                if isinstance(node := config._get_node(key), Node) and not node._is_interpolation():
                    config[key] = _inner_fn(config[key])
        elif isinstance(config, ListConfig):
            config = copy.deepcopy(config)

            for ii in range(len(config)):
                if isinstance(node := config._get_node(ii), Node) and not node._is_interpolation():
                    config[ii] = _inner_fn(config[ii])

        return config

    return _inner_fn


@apply_recursively
def resolve_inherit(config: Any) -> Any:
    """Resolve _inherit_ in config

    Merge the dict in ``_inherit_`` into a dict of the same hierarchy.
    ``_inherit_`` is specified by omegaconf interpolation

    Args:
        config (DictConfig | ListConfig): The configuration loaded by load_config

    Returns:
        DictConfig | ListConfig: The configuration without ``_inherit_``
    """
    if isinstance(config, DictConfig) and "_inherit_" in config:
        inherit_configs = config["_inherit_"]
        if not isinstance(inherit_configs, ListConfig):
            inherit_configs = [inherit_configs]

        config.pop("_inherit_")

        for inherit_config in inherit_configs:
            if isinstance(inherit_config, DictConfig):
                config = oc.merge(inherit_config, config)

    return config


@apply_recursively
def pathlib2str_config(config: Any) -> Any:
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

    if isinstance(config, Path):
        config = str(config)
    
    return config
