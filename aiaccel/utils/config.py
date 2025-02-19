from typing import Any

from copy import deepcopy
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlparse

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
    pip_show_result = subprocess.run(["pip", "show", package_name], capture_output=True, text=True)
    version, location = None, None

    for line in pip_show_result.stdout.splitlines():
        if line.startswith("Version:"):
            version = line.split(": ", 1)[1]
        if line.startswith("Location:"):
            location = line.split(": ", 1)[1]

    if version is not None and location is not None:
        file_name = f"{location}/{package_name}-{version}.dist-info/direct_url.json"
        if os.path.isfile(file_name):
            # read direct_url.json
            with open(file_name) as f:
                dist_info = json.load(f)

            git_url, install_commit_id = None, None
            if "https" in dist_info["url"]:
                # pip install git+https
                install_commit_id = dist_info["vcs_info"]["commit_id"]
                git_url = dist_info["url"]

            elif "file://" in dist_info["url"]:
                # pip install .
                parsed = urlparse(dist_info["url"])
                file_path = Path(unquote(parsed.path))

                result_git_status = subprocess.run(
                    ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=file_path
                )
                install_commit_id = result_git_status.stdout.rstrip("\n")

                result_git_config = subprocess.run(
                    ["git", "config", "--get", "remote.origin.url"], capture_output=True, text=True, cwd=file_path
                )
                git_url = result_git_config.stdout.rstrip("\n")

            if git_url is not None and install_commit_id is not None:
                # get commit id in git
                git_ls_result = subprocess.run(["git", "ls-remote", "--heads", git_url], capture_output=True, text=True)

                # check commit id
                return install_commit_id in git_ls_result.stdout

    return None


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
