# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from pathlib import Path

from omegaconf import OmegaConf as oc  # noqa: N813

import torch

from pytest_mock import MockerFixture

from aiaccel.torch.apps import upload_hf


def test_create_model_card(mocker: MockerFixture, tmp_path: Path) -> None:
    mocker.patch(
        "aiaccel.torch.apps.upload_hf.Prompt.ask",
        side_effect=["test-model", "mit", "audio-classification", "en, ja", "Test model."],
    )
    path = tmp_path / "README.md"

    upload_hf.create_model_card(path, "test/test-model")

    content = path.read_text(encoding="utf-8")
    assert "license: mit" in content
    assert "pipeline_tag: audio-classification" in content
    assert "library_name: aiaccel" in content
    assert "# test-model" in content
    assert "Test model." in content


def test_stage_config_cleaning(mocker: MockerFixture, tmp_path: Path) -> None:
    config_path = tmp_path / "merged_config.yaml"
    target = tmp_path / "hf_upload" / "merged_config.yaml"
    oc.save(
        {
            "checkpoint_filename": "merged",
            "working_directory": "/home/user/train",
            "relative_path": "data/train",
        },
        config_path,
    )

    mocker.patch("aiaccel.torch.apps.upload_hf.Confirm.ask", return_value=True)
    mocker.patch("aiaccel.torch.apps.upload_hf.review_file", return_value=True)

    assert upload_hf.stage_cleaned_file(
        config_path,
        target,
        lambda path: oc.to_container(oc.load(path), resolve=False),
        oc.save,
    )

    staged = oc.to_container(oc.load(target))
    assert staged == {"checkpoint_filename": "merged", "relative_path": "data/train"}


def test_stage_checkpoint_cleaning(mocker: MockerFixture, tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "merged.ckpt"
    target = tmp_path / "hf_upload" / "checkpoints" / "merged.ckpt"
    torch.save(
        {
            "state_dict": {"weight": torch.tensor([1.0])},
            "callbacks": {"dirpath": "/home/user/checkpoints"},
        },
        checkpoint_path,
    )

    mocker.patch("aiaccel.torch.apps.upload_hf.Confirm.ask", return_value=True)
    mocker.patch("aiaccel.torch.apps.upload_hf.review_file", return_value=True)

    assert upload_hf.stage_cleaned_file(
        checkpoint_path,
        target,
        lambda path: torch.load(path, map_location="cpu", weights_only=False),
        torch.save,
    )

    staged = torch.load(target, map_location="cpu", weights_only=False)
    assert "dirpath" not in staged["callbacks"]
    assert torch.equal(staged["state_dict"]["weight"], torch.tensor([1.0]))


def test_existing_staged_file_can_be_reused(mocker: MockerFixture, tmp_path: Path) -> None:
    source = tmp_path / "merged_config.yaml"
    target = tmp_path / "hf_upload" / "merged_config.yaml"
    target.parent.mkdir()
    oc.save({"checkpoint_filename": "merged"}, source)
    oc.save({"checkpoint_filename": "hand-edited"}, target)

    mocker.patch("aiaccel.torch.apps.upload_hf.review_file", return_value=True)
    load = mocker.Mock()
    save = mocker.Mock()

    assert upload_hf.stage_cleaned_file(source, target, load, save)
    assert oc.load(target).checkpoint_filename == "hand-edited"
    load.assert_not_called()
    save.assert_not_called()


def test_upload_model(mocker: MockerFixture, tmp_path: Path) -> None:
    upload_dir = tmp_path / "hf_upload"
    (upload_dir / "checkpoints").mkdir(parents=True)
    (upload_dir / "README.md").write_text("# model\n", encoding="utf-8")
    (upload_dir / "merged_config.yaml").write_text("checkpoint_filename: merged\n", encoding="utf-8")
    (upload_dir / "checkpoints" / "merged.ckpt").touch()

    mocker.patch("aiaccel.torch.apps.upload_hf.Confirm.ask", return_value=True)
    mock_login = mocker.patch("aiaccel.torch.apps.upload_hf.login")
    api = mocker.Mock()
    api.repo_exists.return_value = True
    mocker.patch("aiaccel.torch.apps.upload_hf.HfApi", return_value=api)

    upload_hf.upload_model(upload_dir, "test/model")

    mock_login.assert_called_once_with()
    api.repo_exists.assert_called_once_with(repo_id="test/model", repo_type="model")
    api.create_repo.assert_not_called()
    api.upload_folder.assert_called_once_with(folder_path=upload_dir, repo_id="test/model", repo_type="model")
