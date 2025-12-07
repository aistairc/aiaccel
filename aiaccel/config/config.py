# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing import Any

from collections.abc import Callable
import copy
from importlib import resources
from pathlib import Path
import re

from colorama import Fore
from omegaconf import DictConfig, ListConfig, Node
from omegaconf import OmegaConf as oc  # noqa:N813
from omegaconf._utils import OmegaConfDumper

from simpleeval import simple_eval
import yaml
from yaml.resolver import BaseResolver

_OMEGACONF_INITIALIZED = False


def setup_omegaconf(mode: str = "|") -> None:
    """
    Configure OmegaConf to dump multiline strings consistently and register custom resolvers.

    Args:
        mode (str, optional): YAML style for multiline strings; defaults to block style "|".
    """

    global _OMEGACONF_INITIALIZED
    if _OMEGACONF_INITIALIZED:
        return

    def str_representer(dumper: OmegaConfDumper, data: str) -> yaml.Node:
        style = mode if len(data.splitlines()) > 1 else None
        return dumper.represent_scalar(BaseResolver.DEFAULT_SCALAR_TAG, data, style=style)

    OmegaConfDumper.add_representer(str, str_representer)
    OmegaConfDumper.str_representer_added = True

    oc.register_new_resolver("eval", simple_eval, replace=True)
    oc.register_new_resolver("resolve_pkg_path", resources.files, replace=True)

    _OMEGACONF_INITIALIZED = True


def prepare_config(
    config_filename: str | Path,
    working_directory: str | Path | None = None,
    overwrite_config: DictConfig | ListConfig | dict[Any, Any] | list[Any] | None = None,
    *,
    print_config: bool = False,
    save_config: bool = False,
    save_directory: str | Path | None = None,
    save_filename: str = "merged_config.yaml",
    load_config_kwargs: dict[str, Any] | None = None,
    print_config_kwargs: dict[str, Any] | None = None,
) -> DictConfig | ListConfig:
    """Assemble a fully resolved configuration built on top of :func:`load_config`.

    This helper first delegates to :func:`load_config` to read the YAML file (optionally using
    ``load_config_kwargs``), then merges ``overwrite_config``, resolves inheritance, and optionally prints
    or stores the merged configuration. Use this function when you need a ready-to-use configuration object.

    Args:
        config_filename (str | Path):
            Path to the configuration
        working_directory (str | Path | None, optional):
            Path to the working directory to store in merge_user_config.
            If None, the parent directory of config_filename is used.
            Defaults to None.
        overwrite_config (DictConfig | ListConfig | dict[Any, Any] | list[Any] | None, optional):
            Configuration that overwrites the loaded configuration before resolving.
            Defaults to None.
        print_config (bool, optional):
            Print the merged configuration immediately after loading.
            Defaults to False.
        save_config (bool, optional):
            Save the resolved configuration immediately after loading.
            Defaults to False.
        save_directory (str | Path | None, optional):
            Directory to store the saved configuration. If omitted, the configuration's
            ``working_directory`` is used.
        save_filename (str, optional):
            Filename used when ``save_config`` is True. Defaults to ``"merged_config.yaml"``.
        load_config_kwargs (dict[str, Any] | None, optional):
            Keyword arguments forwarded to :func:`load_config`. Use this to override ``parent_config`` or any
            future options when loading the base YAML.
        print_config_kwargs (dict[str, Any] | None, optional):
            Keyword arguments forwarded to :func:`print_config` when ``print_config`` is True.
            Allows overriding ``line_length`` or ``print`` built-in parameters (e.g., ``file``).

    Returns:
        DictConfig | ListConfig:
            The fully resolved configuration. Use ``print_config=True`` or ``save_config=True`` to
            dump intermediate states.
    """
    setup_omegaconf()

    working_directory = (
        str(working_directory) if working_directory is not None else str(Path(config_filename).parent.resolve())
    )

    # Load config
    load_config_kwargs = copy.deepcopy(load_config_kwargs or {})
    parent_config = load_config_kwargs.pop("parent_config", oc.create({}))
    if not isinstance(parent_config, (DictConfig, ListConfig, Node)):
        parent_config = oc.create(parent_config)
    default_parent_config = oc.create(
        {
            "config_path": str(config_filename),
            "working_directory": str(working_directory)
            if working_directory is not None
            else str(Path(config_filename).parent.resolve()),
        }
    )
    effective_parent_config = oc.merge(default_parent_config, parent_config)
    config = oc.merge(
        load_config(
            config_filename,
            parent_config=effective_parent_config,
            **load_config_kwargs,
        ),
        overwrite_config if overwrite_config is not None else oc.create({}),  # Overwrite loaded config
    )

    if print_config:
        print_config_fn = globals()["print_config"]
        print_config_fn(config, **(print_config_kwargs or {}))

    config = resolve_inherit(config)

    if save_config:
        target_directory = Path(save_directory) if save_directory is not None else Path(config.working_directory)
        target_directory.mkdir(parents=True, exist_ok=True)

        with open(target_directory / save_filename, "w") as f:
            oc.save(pathlib2str_config(config), f)

    return config


def load_config(
    config_filename: str | Path,
    parent_config: dict[str, Any] | DictConfig | ListConfig | None = None,
) -> DictConfig | ListConfig:
    """Load YAML configuration

    When the user specifies ``_base_``, the specified YAML file is loaded as the base,
    and the original configuration is merged with the base config.
    If the configuration specified in ``_base_`` also contains ``_base_``, the process is handled recursively.

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


def print_config(
    config: ListConfig | DictConfig,
    line_length: int = 80,
    **print_kwargs: Any,
) -> None:
    """
    Print the given configuration with syntax highlighting.

    This function converts `pathlib.Path` objects to strings before printing,
    ensuring that the output YAML format remains valid. It also highlights
    configuration keys in yellow for better readability.

    Args:
        config (ListConfig | DictConfig): The configuration to print.
        line_length (int, optional): The width of the separator line (default: 80).

    """
    setup_omegaconf()

    config = pathlib2str_config(config)  # https://github.com/omry/omegaconf/issues/82
    print_kwargs = print_kwargs or {}

    print("=" * line_length, **print_kwargs)
    for line in oc.to_yaml(config).splitlines():
        print(re.sub(r"(\s*)(\w+):", rf"\1{Fore.YELLOW}\2{Fore.RESET}:", line, count=1), **print_kwargs)
    print("=" * line_length, **print_kwargs)


def apply_recursively(
    func: Callable[[DictConfig | ListConfig], DictConfig | ListConfig],
) -> Callable[[DictConfig | ListConfig], DictConfig | ListConfig]:
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
                if (
                    isinstance(node_dict := config._get_node(key), Node)
                    and not node_dict._is_interpolation()
                    and isinstance(config[key], DictConfig | ListConfig)
                ):
                    config[key] = _inner_fn(config[key])
        elif isinstance(config, ListConfig):
            config = copy.deepcopy(config)

            for ii in range(len(config)):
                if (
                    isinstance(node_list := config._get_node(ii), Node)
                    and not node_list._is_interpolation()
                    and isinstance(config[ii], DictConfig | ListConfig)
                ):
                    config[ii] = _inner_fn(config[ii])

        return config

    return _inner_fn


@apply_recursively
def resolve_inherit(config: DictConfig | ListConfig) -> DictConfig | ListConfig:
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
def pathlib2str_config(config: DictConfig | ListConfig) -> DictConfig | ListConfig:
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

    config = copy.deepcopy(config)

    if isinstance(config, DictConfig):
        for k, v in config.items():
            if isinstance(v, Path):
                config[k] = str(v)
    elif isinstance(config, ListConfig):
        for ii, value in enumerate(config):
            if isinstance(value, Path):
                config[ii] = str(value)

    return config
