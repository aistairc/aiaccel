from pathlib import Path
from unittest.mock import Mock, patch

from aiaccel.config.apps import check_git
from aiaccel.config.git import PackageGitStatus


def test_check_git() -> None:
    mock_args = Mock()
    mock_args.config = str(Path(__file__).parent / "test_check_git.yaml")

    # Success
    with (
        patch("aiaccel.config.apps.check_git.collect_git_status_from_config", return_value=[]),
        patch("argparse.ArgumentParser.parse_known_args", return_value=(mock_args, [])),
    ):
        check_git.main()

    # Failed
    with (
        patch(
            "aiaccel.config.apps.check_git.collect_git_status_from_config",
            return_value=[PackageGitStatus("test_package", "test_id", [])],
        ),
        patch("argparse.ArgumentParser.parse_known_args", return_value=(mock_args, [])),
    ):
        try:
            check_git.main()
        except SystemExit:
            pass
        else:
            raise AssertionError()
