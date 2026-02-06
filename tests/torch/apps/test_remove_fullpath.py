# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
import os
from pathlib import Path
import subprocess

from omegaconf import OmegaConf as oc  # noqa: N813

import torch

import pytest


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


def test_merge_model(workspace_factory: Callable[..., AbstractContextManager[Path]]) -> None:
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

        # execute merge_model
        cmd = (
            "aiaccel-torch remove-fullpath "
            f"--config_path {workspace}/config.yaml "
            "--save_config_filename pathremoved_config.yaml "
            f"--ckpt_path {workspace}/checkpoints/merged.ckpt "
            "--save_ckpt_filename pathremoved.ckpt "
        )
        subprocess.run(cmd, shell=True, check=True)

        # test
        pathremoved_ckpt = torch.load(workspace / "checkpoints" / "pathremoved.ckpt", map_location="cpu")
        assert "Fullpath" not in pathremoved_ckpt

        pathremoved_config = oc.to_container(oc.load(workspace / "pathremoved_config.yaml"))
        assert isinstance(pathremoved_config, dict)
        assert "Fullpath" not in pathremoved_config
