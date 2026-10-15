# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing_extensions import Self

from pathlib import Path
from unittest.mock import patch
import warnings

from omegaconf import OmegaConf

from aiaccel.torch.lightning.ckpt import get_checkpoint_path, load_checkpoint


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


def test_get_checkpoint_path() -> None:
    assert get_checkpoint_path("model", "last") == Path("model/checkpoints/last.ckpt")
    assert get_checkpoint_path("model", "last.ckpt") == Path("model/checkpoints/last.ckpt")


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


def test_load_checkpoint_merges_loading_options(tmp_path: Path) -> None:
    config = OmegaConf.create(
        {
            "checkpoint_filename": "model",
            "task": {
                "_target_": f"{__name__}.DummyModel",
                "strict": False,
                "weights_only": True,
            },
        }
    )
    overwrite_config = {"task": {"custom_setting": "keep"}}
    with (
        patch("aiaccel.torch.lightning.ckpt.prepare_config", return_value=config) as prepare,
        patch("aiaccel.torch.lightning.ckpt.instantiate", return_value=DummyModel()) as instantiate,
    ):
        load_checkpoint(
            tmp_path,
            device="cpu",
            overwrite_config=overwrite_config,
            strict=False,
            weights_only=True,
        )

    merged = prepare.call_args.kwargs["overwrite_config"]
    assert OmegaConf.to_container(merged) == {"task": {"custom_setting": "keep", "strict": False, "weights_only": True}}
    assert overwrite_config == {"task": {"custom_setting": "keep"}}
    assert instantiate.call_args.kwargs["map_location"] == "cpu"
    assert instantiate.call_args.args[0].strict is False
    assert instantiate.call_args.args[0].weights_only is True


def test_load_checkpoint_suppresses_only_key_warnings(tmp_path: Path) -> None:
    config = OmegaConf.create(
        {"checkpoint_filename": "model", "task": {"_target_": f"{__name__}.DummyModel", "strict": False}}
    )

    def instantiate_with_warnings(*args: object, **kwargs: object) -> DummyModel:
        warnings.warn(
            "Found keys that are in the model state dict but not in the checkpoint: ['weight']",
            UserWarning,
            stacklevel=1,
        )
        warnings.warn(
            "A different warning",
            UserWarning,
            stacklevel=1,
        )

        return DummyModel()

    with (
        patch("aiaccel.torch.lightning.ckpt.prepare_config", return_value=config),
        patch("aiaccel.torch.lightning.ckpt.instantiate", side_effect=instantiate_with_warnings),
        patch("aiaccel.torch.lightning.ckpt.logger.warning") as warning_log,
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        load_checkpoint(tmp_path, strict=False)

    assert [str(item.message) for item in caught] == ["A different warning"]
    warning_log.assert_called_once_with("Checkpoint loading with strict=%s.", False)


def test_load_checkpoint_warns_when_strict_true(tmp_path: Path) -> None:
    config = OmegaConf.create(
        {"checkpoint_filename": "model", "task": {"_target_": f"{__name__}.DummyModel", "strict": True}}
    )
    with (
        patch("aiaccel.torch.lightning.ckpt.prepare_config", return_value=config),
        patch("aiaccel.torch.lightning.ckpt.instantiate", return_value=DummyModel()),
        patch("aiaccel.torch.lightning.ckpt.logger.warning") as warning_log,
    ):
        load_checkpoint(tmp_path, strict=True)

    warning_log.assert_called_once_with("Checkpoint loading with strict=%s.", True)
