# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from pathlib import Path

from omegaconf import OmegaConf as oc  # noqa: N813

import torch

import pytest
from pytest_mock import MockerFixture

from aiaccel.torch.apps import upload_hf
from aiaccel.torch.utils.remove_absolute_path import is_absolute_path, remove_absolute_path


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
    assert is_absolute_path(path) is expected


def test_remove_absolute_path() -> None:
    obj = {
        "relative": "checkpoints/model.ckpt",
        "absolute": "/home/user/model.ckpt",
        "nested": {"data_dir": "/data/dataset", "value": 1},
        "items": ["keep", "/remove/me"],
    }

    cleaned, removed = remove_absolute_path(obj)

    assert cleaned == {
        "relative": "checkpoints/model.ckpt",
        "nested": {"value": 1},
        "items": ["keep"],
    }
    assert removed == ["absolute", "nested.data_dir", "items[1]"]


def test_resolve_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "checkpoints").mkdir()
    checkpoint_path = tmp_path / "checkpoints" / "merged.ckpt"
    checkpoint_path.touch()
    config = oc.create({"checkpoint_filename": "merged"})

    assert upload_hf.resolve_checkpoint(tmp_path, config) == checkpoint_path


def test_stage_config_cleaning(mocker: MockerFixture, tmp_path: Path) -> None:
    config_path = tmp_path / "merged_config.yaml"
    upload_dir = tmp_path / "hf_upload"
    upload_dir.mkdir()
    oc.save(
        {
            "checkpoint_filename": "merged",
            "working_directory": "/home/user/train",
            "relative_path": "data/train",
        },
        config_path,
    )

    mocker.patch("aiaccel.torch.apps.upload_hf.yes_no_input", side_effect=[True, True])
    mocker.patch("aiaccel.torch.apps.upload_hf.wait_for_review")

    assert upload_hf.stage_config(config_path, upload_dir) is True

    staged = oc.to_container(oc.load(upload_dir / "merged_config.yaml"))
    assert staged == {"checkpoint_filename": "merged", "relative_path": "data/train"}


def test_stage_checkpoint_cleaning(mocker: MockerFixture, tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "merged.ckpt"
    upload_dir = tmp_path / "hf_upload"
    upload_dir.mkdir()
    torch.save(
        {
            "state_dict": {"weight": torch.tensor([1.0])},
            "callbacks": {"dirpath": "/home/user/checkpoints"},
        },
        checkpoint_path,
    )

    mocker.patch("aiaccel.torch.apps.upload_hf.yes_no_input", side_effect=[True, True])
    mocker.patch("aiaccel.torch.apps.upload_hf.wait_for_review")

    assert upload_hf.stage_checkpoint(checkpoint_path, upload_dir) is True

    staged = torch.load(upload_dir / "checkpoints" / "merged.ckpt", map_location="cpu", weights_only=False)
    assert "dirpath" not in staged["callbacks"]
    assert torch.equal(staged["state_dict"]["weight"], torch.tensor([1.0]))


def test_existing_staged_config_can_be_reused(mocker: MockerFixture, tmp_path: Path) -> None:
    config_path = tmp_path / "merged_config.yaml"
    upload_dir = tmp_path / "hf_upload"
    upload_dir.mkdir()
    oc.save({"checkpoint_filename": "merged"}, config_path)
    staged_path = upload_dir / "merged_config.yaml"
    oc.save({"checkpoint_filename": "hand-edited"}, staged_path)

    mocker.patch("aiaccel.torch.apps.upload_hf.yes_no_input", return_value=True)
    mock_wait = mocker.patch("aiaccel.torch.apps.upload_hf.wait_for_review")

    assert upload_hf.stage_config(config_path, upload_dir) is True
    assert oc.load(staged_path).checkpoint_filename == "hand-edited"
    mock_wait.assert_called_once_with(staged_path)


def test_upload_model(mocker: MockerFixture, tmp_path: Path) -> None:
    upload_dir = tmp_path / "hf_upload"
    (upload_dir / "checkpoints").mkdir(parents=True)
    (upload_dir / "README.md").write_text("# model\n", encoding="utf-8")
    (upload_dir / "merged_config.yaml").write_text("checkpoint_filename: merged\n", encoding="utf-8")
    (upload_dir / "checkpoints" / "merged.ckpt").touch()

    mocker.patch("aiaccel.torch.apps.upload_hf.yes_no_input", return_value=True)
    mock_login = mocker.patch("aiaccel.torch.apps.upload_hf.login")
    api = mocker.Mock()
    api.repo_exists.return_value = True
    mocker.patch("aiaccel.torch.apps.upload_hf.HfApi", return_value=api)

    upload_hf.upload_model(upload_dir, "test/model")

    mock_login.assert_called_once_with()
    api.repo_exists.assert_called_once_with(repo_id="test/model", repo_type="model")
    api.create_repo.assert_not_called()
    api.upload_folder.assert_called_once_with(folder_path=upload_dir, repo_id="test/model", repo_type="model")
