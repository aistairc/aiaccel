# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, contextmanager
import os
from pathlib import Path
import subprocess

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
        # Setup ckpt for test
        ckpt = {}
        ckpt["state_dict"] = {"layer1": 0.1, "layer2": 1.0}
        torch.save(ckpt, workspace / "checkpoints" / "epoch=0001_score=0.1.ckpt")

        ckpt["state_dict"] = {"layer1": 0.2, "layer2": 2.0}
        torch.save(ckpt, workspace / "checkpoints" / "epoch=0002_score=0.2.ckpt")

        ckpt["state_dict"] = {"layer1": 0.3, "layer2": 3.0}
        torch.save(ckpt, workspace / "checkpoints" / "epoch=0003_score=0.3.ckpt")

        # execute merge_model
        merged_ckpt_name = "merged.ckpt"
        cmd = (
            f"aiaccel-torch merge-model {workspace} "
            f"--ckpt_name {merged_ckpt_name} "
            "--n_ckpt 2 "
            "--direction max "
            "--reqexp 'score=([0-9.]+)'"
        )
        subprocess.run(cmd, shell=True, check=True)

        # test
        merged_ckpt = torch.load(workspace / "checkpoints" / merged_ckpt_name, map_location="cpu")
        assert merged_ckpt["state_dict"]["layer1"] == 0.25
        assert merged_ckpt["state_dict"]["layer2"] == 2.5
