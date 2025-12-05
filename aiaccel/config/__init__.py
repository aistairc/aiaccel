from aiaccel.config.config import (
    load_config,
    pathlib2str_config,
    print_config,
    resolve_inherit,
)
from aiaccel.config.git import PackageGitStatus, collect_git_status_from_config, print_git_status

__all__ = [
    "load_config",
    "pathlib2str_config",
    "print_config",
    "resolve_inherit",
    "PackageGitStatus",
    "collect_git_status_from_config",
    "print_git_status",
]

from importlib import resources

from omegaconf import OmegaConf as oc  # noqa:N813
from omegaconf._utils import OmegaConfDumper

from simpleeval import simple_eval
import yaml
from yaml.resolver import BaseResolver

IS_NOT_INIT = True


def setup_omegaconf(mode: str = "|") -> None:
    """
    Overwrites the default string representation in OmegaConf's YAML dumper.
    And, register custom resolvers "eval" and "resolve_pkg_path"

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

    # Register custom resolvers
    oc.register_new_resolver("eval", simple_eval)
    oc.register_new_resolver("resolve_pkg_path", resources.files)

    global IS_NOT_INIT
    IS_NOT_INIT = False
