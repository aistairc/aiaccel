# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from pathlib import Path

from pytest_mock import MockerFixture

from aiaccel.config.apps import check_git
from aiaccel.config.git import PackageGitStatus


def test_check_git(mocker: MockerFixture) -> None:
    mock_args = mocker.Mock()
    mock_args.config = str(Path(__file__).parent / "test_check_git.yaml")

    mock_argparse = mocker.patch("argparse.ArgumentParser.parse_known_args")
    mock_argparse.return_value = (mock_args, [])

    # Success
    mock_func = mocker.patch("aiaccel.config.apps.check_git.collect_git_status_from_config")
    mock_func.return_value = []

    try:
        check_git.main()
    except SystemExit as e:
        if e.code != 0:
            raise AssertionError() from e

    # Failed
    mock_func = mocker.patch("aiaccel.config.apps.check_git.collect_git_status_from_config")
    mock_func.return_value = [PackageGitStatus("test_package", "test_id", [])]

    try:
        check_git.main()
    except SystemExit as e:
        if e.code != 1:
            raise AssertionError() from e
    else:
        raise AssertionError()
