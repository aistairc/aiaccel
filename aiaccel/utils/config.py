from typing import Any

from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import re
import subprocess

from colorama import Fore
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as oc  # noqa:N813


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

    if parent_config is None:
        parent_config = {}

    config = oc.merge(oc.load(config_filename), parent_config)

    if isinstance(config, DictConfig) and "_base_" in config:
        base_paths = config["_base_"]
        if not isinstance(base_paths, ListConfig):
            base_paths = [base_paths]

        config.pop("_base_")
        for base_path in base_paths:
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


def check_commit(package_name: str) -> bool | None:
    # get package location
    spec = importlib.util.find_spec(package_name)

    if spec is None:
        return None

    if spec.origin is not None:
        module_path = Path(spec.origin).parent
    elif spec.submodule_search_locations is not None:
        module_path = Path(os.path.abspath(spec.submodule_search_locations[0]))
    else:
        return None

    # get repository path
    result = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=module_path, capture_output=True, text=True)
    try:
        result.check_returncode()
    except subprocess.CalledProcessError:
        return None

    repository_path = result.stdout.splitlines()[0]

    # check git status
    result = subprocess.run(["git", "status", "-s"], cwd=repository_path, capture_output=True, text=True)
    status = result.stdout.splitlines()

    return len(status) == 0


def get_target_module(config: ListConfig | DictConfig) -> list[str]:
    target_module = []

    if isinstance(config, DictConfig):
        for key, value in config.items():
            if key == "_target_":
                target_module.append(value)
            target_module += get_target_module(value)
    elif isinstance(config, ListConfig):
        for item in config:
            target_module += get_target_module(item)

    return target_module


def check_commit_target_modules(config: DictConfig | ListConfig) -> dict[str, bool | None]:
    check_commit_dict = {}

    for target in get_target_module(config):
        package_name = target.split(".")[0]
        if package_name not in check_commit_dict:
            check_commit_dict[package_name] = check_commit(package_name)

    return check_commit_dict
