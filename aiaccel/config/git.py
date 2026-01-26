# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import subprocess

from omegaconf import DictConfig, ListConfig

__all__ = [
    "PackageGitStatus",
    "collect_git_status_from_config",
    "print_git_status",
]


@dataclass
class PackageGitStatus:
    """
    Represents the Git status of a package.

    Attributes:
        package_name (str): The name of the package.
        commit_id (str): The current Git commit ID of the repository.
        status (list[str]): A list of uncommitted files in the repository.
    """

    package_name: str
    commit_id: str
    status: list[str]

    def ready(self) -> bool:
        """
        Determines if there are no uncommitted changes.

        Returns:
            bool: True if there are no uncommitted files, otherwise False.
        """

        return len(self.status) == 0


def collect_git_status_from_config(config: DictConfig | ListConfig) -> list[PackageGitStatus]:
    """
    Collects the Git status of packages specified in the given configuration.

    Args:
        config (DictConfig | ListConfig): The configuration containing package references.

    Returns:
        list[PackageGitStatus]: A list of `PackageGitStatus` objects representing
                                the Git status of the detected packages.
    """

    status_list = []

    package_names = collect_target_packages(config)
    package_names.sort()

    for package_name in package_names:
        status = get_git_status(package_name)

        if status is not None:
            status_list.append(status)

    return status_list


def print_git_status(status: PackageGitStatus | list[PackageGitStatus]) -> None:
    """
    Prints the Git status of a package or a list of packages.

    Args:
        status (PackageGitStatus | list[PackageGitStatus]): The Git status to print.
    """

    status_list = status if isinstance(status, list) else [status]

    for status in status_list:
        print(f"{status.package_name} @ {status.commit_id}")
        for st in status.status:
            print(f"  {st}")


def get_git_status(package_name: str) -> PackageGitStatus | None:
    """
    Retrieves the Git status of a given package.

    Args:
        package_name (str): The name of the package to check.

    Returns:
        PackageGitStatus | None: A `PackageGitStatus` object if the package is found
                                 and under Git control, otherwise None.
    """

    # get package location
    spec = importlib.util.find_spec(package_name)

    if spec is None:
        return None

    if spec.origin is not None:
        module_path = Path(spec.origin).parent.resolve()
    elif spec.submodule_search_locations is not None:
        module_path = Path(os.path.abspath(spec.submodule_search_locations[0])).resolve()
    else:
        return None

    # get repository path
    result = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=module_path, capture_output=True, text=True)
    if result.returncode != 0:
        return None

    repository_path = Path(result.stdout.splitlines()[0]).resolve()

    # check git_ignore
    result = subprocess.run(["git", "check-ignore", module_path], cwd=repository_path, capture_output=True, text=True)
    if result.returncode == 0:
        return None

    # get commit id
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository_path, capture_output=True, text=True)
    commit_id = result.stdout.splitlines()[0]

    # check git status
    result = subprocess.run(["git", "status", "-s"], cwd=repository_path, capture_output=True, text=True)
    status = result.stdout.splitlines()

    return PackageGitStatus(package_name, commit_id, status)


def collect_target_packages(config: ListConfig | DictConfig) -> list[str]:
    """
    Extracts the names of target packages from the given configuration.

    Args:
        config (ListConfig | DictConfig): The configuration to process.

    Returns:
        list[str]: A list of package names extracted from the configuration.
    """

    target_packages = set()

    def inner_func(_config: ListConfig | DictConfig) -> None:
        if isinstance(_config, DictConfig):
            for key, value in _config.items():
                if key == "_target_":
                    package_name, *_ = value.split(".")
                    target_packages.add(package_name)

                inner_func(value)

        elif isinstance(_config, ListConfig):
            for item in _config:
                inner_func(item)

    inner_func(config)

    return list(target_packages)
