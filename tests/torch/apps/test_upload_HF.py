# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
import os
from pathlib import Path

from omegaconf import OmegaConf as oc  # noqa: N813

import torch

import pytest
from pytest_mock import MockerFixture

from aiaccel.torch.apps import upload_huggingface


@pytest.fixture()
def workspace_factory(
    tmp_path_factory: pytest.TempPathFactory,
) -> Callable[[], AbstractContextManager[Path]]:
    @contextmanager
    def _factory() -> Generator[Path, None, None]:
        tmp_path = tmp_path_factory.mktemp("workspace")
        org_path = Path.cwd()

        os.mkdir(tmp_path / "checkpoints")

        try:
            os.chdir(tmp_path)
            yield tmp_path
        finally:
            os.chdir(org_path)

    return _factory


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/home/user/model.ckpt", True),
        (r"C:\Users\user\model.ckpt", True),
        (r"\\server\share\model.ckpt", True),
        ("checkpoints/model.ckpt", False),
    ],
)
def test_is_absolute_path(path: str, expected: bool) -> None:
    assert upload_huggingface.is_absolute_path(path) is expected


def test_remove_fullpath(mocker: MockerFixture, workspace_factory: Callable[[], AbstractContextManager[Path]]) -> None:
    with workspace_factory() as workspace:
        # Setup ckpt and config for test
        ckpt = {}
        ckpt["state_dict"] = {"layer1": 0.1, "layer2": 1.0, "FullPath1": "/full/path/test.txt"}
        torch.save(ckpt, workspace / "checkpoints" / "merged.ckpt")

        config = {
            "NotFullPath": "notfullpath",
            "FullPath": "/full/path/test.txt",
        }
        with open(workspace / "config.yaml", "w") as f:
            oc.save(config, f)

        # execute upload-huggingface
        mocker.patch(
            "sys.argv",
            [
                "upload-huggingface",
                "--config_path",
                str(workspace / "config.yaml"),
                "--save_config_filename",
                "pathremoved_config.yaml",
                "--ckpt_path",
                str(workspace / "checkpoints" / "merged.ckpt"),
                "--save_ckpt_filename",
                "pathremoved.ckpt",
            ],
        )

        upload_huggingface.main()

        # test
        pathremoved_ckpt = torch.load(workspace / "checkpoints" / "pathremoved.ckpt", map_location="cpu")
        assert "FullPath1" not in pathremoved_ckpt["state_dict"]
        assert pathremoved_ckpt["state_dict"]["layer1"] == 0.1
        assert pathremoved_ckpt["state_dict"]["layer2"] == 1.0

        pathremoved_config = oc.to_container(oc.load(workspace / "pathremoved_config.yaml"))
        assert isinstance(pathremoved_config, dict)
        assert "FullPath" not in pathremoved_config
        assert pathremoved_config["NotFullPath"] == "notfullpath"


def test_upload_huggingface(
    mocker: MockerFixture, workspace_factory: Callable[[], AbstractContextManager[Path]]
) -> None:
    with workspace_factory() as workspace:
        # Setup
        ckpt = {"state_dict": {"layer1": 0.1}}
        torch.save(ckpt, workspace / "checkpoints" / "merged.ckpt")

        config = {"foo": "bar"}
        with open(workspace / "config.yaml", "w") as f:
            oc.save(config, f)

        mocker.patch(
            "sys.argv",
            [
                "upload_huggingface",
                "--config_path",
                str(workspace / "config.yaml"),
                "--ckpt_path",
                str(workspace / "checkpoints" / "merged.ckpt"),
                "--repo_id",
                "test/model",
                "--repo_type",
                "model",
            ],
        )

        mock_yes_no = mocker.patch(
            "aiaccel.torch.apps.upload_huggingface.yes_no_input",
            return_value=True,
        )
        mock_login = mocker.patch("aiaccel.torch.apps.upload_huggingface.login")
        mock_repo_exists = mocker.patch(
            "aiaccel.torch.apps.upload_huggingface.repo_exists",
            return_value=True,
        )
        mock_upload_file = mocker.patch("aiaccel.torch.apps.upload_huggingface.upload_file")

        upload_huggingface.main()

        mock_yes_no.assert_called_once()
        mock_login.assert_called_once()
        mock_repo_exists.assert_called_once_with(
            repo_id="test/model",
            repo_type="model",
        )

        mock_upload_file.assert_any_call(
            path_or_fileobj=workspace / "pathremoved_config.yaml",
            path_in_repo="pathremoved_config.yaml",
            repo_id="test/model",
            repo_type="model",
        )

        mock_upload_file.assert_any_call(
            path_or_fileobj=workspace / "checkpoints" / "pathremoved.ckpt",
            path_in_repo="checkpoints/pathremoved.ckpt",
            repo_id="test/model",
            repo_type="model",
        )
