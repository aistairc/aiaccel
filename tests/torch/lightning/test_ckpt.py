# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing_extensions import Self

from pathlib import Path
from unittest.mock import patch

from omegaconf import OmegaConf

from aiaccel.torch.lightning.ckpt import load_checkpoint


class DummyModel:
    loaded_checkpoint_path: Path | None = None
    loaded_map_location: str | None = None

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: str | Path, map_location: str) -> Self:
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        cls.loaded_checkpoint_path = checkpoint_path
        cls.loaded_map_location = map_location

        return cls()


def test_load_checkpoint_filename_has_no_extension(tmp_path: Path) -> None:
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()

    checkpoint_path = checkpoints_dir / "last.ckpt"
    checkpoint_path.write_text("dummy checkpoint")

    config = OmegaConf.create(
        {
            "checkpoint_filename": "last",
            "task": {
                "_target_": f"{__name__}.DummyModel",
            },
        }
    )

    with patch("aiaccel.torch.lightning.ckpt.prepare_config", return_value=config):
        model, returned_config = load_checkpoint(tmp_path, device="cpu")

    assert isinstance(model, DummyModel)
    assert returned_config is config
    assert DummyModel.loaded_checkpoint_path == checkpoint_path
    assert DummyModel.loaded_map_location == "cpu"
