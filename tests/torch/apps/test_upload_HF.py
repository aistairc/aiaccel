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


def test_remove_fullpath(mocker: MockerFixture, workspace_factory: Callable[..., AbstractContextManager[Path]]) -> None:
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
        assert "FullPath1" not in pathremoved_ckpt

        pathremoved_config = oc.to_container(oc.load(workspace / "pathremoved_config.yaml"))
        assert isinstance(pathremoved_config, dict)
        assert "FullPath" not in pathremoved_config
        assert pathremoved_config["NotFullPath"] == "notfullpath"
