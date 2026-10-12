# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from pathlib import Path

from omegaconf import OmegaConf as oc  # noqa: N813

import torch

from pytest_mock import MockerFixture

from aiaccel.torch.apps import upload_hf


def test_stage_readme_creates_model_card(mocker: MockerFixture, tmp_path: Path) -> None:
    mocker.patch(
        "aiaccel.torch.apps.upload_hf.ask_text",
        side_effect=["test-model", "mit", "test-library", "en, ja", "Test model."],
    )
    mocker.patch("aiaccel.torch.apps.upload_hf.ask_confirm", return_value=True)

    mocker.patch("aiaccel.torch.apps.upload_hf.questionary.select").return_value.unsafe_ask.return_value = (
        "audio-classification"
    )

    model_dir = tmp_path / "model"
    upload_dir = model_dir / "hf_upload"
    model_dir.mkdir()
    upload_dir.mkdir()

    assert upload_hf.stage_readme(model_dir, upload_dir, "test/test-model")

    path = model_dir / "README.md"
    content = path.read_text(encoding="utf-8")
    assert "license: mit" in content
    assert "pipeline_tag: audio-classification" in content
    assert "library_name: test-library" in content
    assert "# test-model" in content
    assert "Test model." in content
    assert (upload_dir / "README.md").read_text(encoding="utf-8") == content


def test_stage_readme_keeps_existing_model_readme(mocker: MockerFixture, tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    upload_dir = model_dir / "hf_upload"
    upload_dir.mkdir(parents=True)
    source = model_dir / "README.md"
    source.write_text("# Original model card", encoding="utf-8")
    confirm = mocker.patch("aiaccel.torch.apps.upload_hf.ask_confirm", return_value=False)
    prompt = mocker.patch("aiaccel.torch.apps.upload_hf.ask_text")

    assert upload_hf.stage_readme(model_dir, upload_dir, "test/model")

    confirm.assert_called_once_with("Regenerate model README.md?", default=False)
    prompt.assert_not_called()
    assert source.read_text(encoding="utf-8") == "# Original model card"
    assert (upload_dir / "README.md").read_text(encoding="utf-8") == "# Original model card"


def test_stage_readme_regenerates_existing_model_readme(mocker: MockerFixture, tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    upload_dir = model_dir / "hf_upload"
    upload_dir.mkdir(parents=True)
    source = model_dir / "README.md"
    source.write_text("# Old model card", encoding="utf-8")
    confirm = mocker.patch("aiaccel.torch.apps.upload_hf.ask_confirm", return_value=True)
    mocker.patch(
        "aiaccel.torch.apps.upload_hf.ask_text",
        side_effect=["new-model", "mit", "aiaccel", "", "New description."],
    )

    mocker.patch("aiaccel.torch.apps.upload_hf.questionary.select").return_value.unsafe_ask.return_value = "skip"
    assert upload_hf.stage_readme(model_dir, upload_dir, "test/model")

    confirm.assert_called_once_with("Regenerate model README.md?", default=False)
    content = source.read_text(encoding="utf-8")
    assert "# new-model" in content
    assert "New description." in content
    assert (upload_dir / "README.md").read_text(encoding="utf-8") == content


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

    mocker.patch("aiaccel.torch.apps.upload_hf.ask_confirm", return_value=True)

    upload_hf.stage_cleaned_file(
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

    mocker.patch("aiaccel.torch.apps.upload_hf.ask_confirm", return_value=True)

    upload_hf.stage_cleaned_file(
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

    mocker.patch("aiaccel.torch.apps.upload_hf.ask_confirm", return_value=False)
    load = mocker.Mock()
    save = mocker.Mock()

    upload_hf.stage_cleaned_file(source, target, load, save)
    assert oc.load(target).checkpoint_filename == "hand-edited"
    load.assert_not_called()
    save.assert_not_called()


def test_upload_model(mocker: MockerFixture, tmp_path: Path) -> None:
    upload_dir = tmp_path / "hf_upload"
    (upload_dir / "checkpoints").mkdir(parents=True)
    (upload_dir / "README.md").write_text("# model\n", encoding="utf-8")
    (upload_dir / "merged_config.yaml").write_text("checkpoint_filename: merged\n", encoding="utf-8")
    (upload_dir / "checkpoints" / "merged.ckpt").touch()

    mocker.patch("aiaccel.torch.apps.upload_hf.ask_confirm", return_value=True)
    mocker.patch("aiaccel.torch.apps.upload_hf.validate_readme", return_value=True)
    mock_login = mocker.patch("aiaccel.torch.apps.upload_hf.login")
    api = mocker.Mock()
    api.repo_exists.return_value = True
    mocker.patch("aiaccel.torch.apps.upload_hf.HfApi", return_value=api)

    upload_hf.upload_model(upload_dir, "test/model")

    mock_login.assert_called_once_with()
    api.repo_exists.assert_called_once_with(repo_id="test/model", repo_type="model")
    api.create_repo.assert_not_called()
    api.upload_folder.assert_called_once_with(folder_path=upload_dir, repo_id="test/model", repo_type="model")


def test_prepare_upload_dir_keeps_files_by_default(mocker: MockerFixture, tmp_path: Path) -> None:
    upload_dir = tmp_path / "hf_upload"
    upload_dir.mkdir()
    existing = upload_dir / "existing.ckpt"
    existing.write_bytes(b"checkpoint")
    confirm = mocker.patch("aiaccel.torch.apps.upload_hf.ask_confirm", return_value=False)

    upload_hf.prepare_upload_dir(upload_dir)

    assert existing.read_bytes() == b"checkpoint"
    confirm.assert_called_once_with("Delete existing hf_upload?", default=False)


def test_prepare_upload_dir_deletes_when_approved(mocker: MockerFixture, tmp_path: Path) -> None:
    upload_dir = tmp_path / "hf_upload"
    upload_dir.mkdir()
    (upload_dir / "old.ckpt").touch()
    mocker.patch("aiaccel.torch.apps.upload_hf.ask_confirm", return_value=True)

    upload_hf.prepare_upload_dir(upload_dir)

    assert upload_dir.is_dir()
    assert list(upload_dir.iterdir()) == []


def test_stage_readme_reuses_existing_file(mocker: MockerFixture, tmp_path: Path) -> None:
    upload_dir = tmp_path / "hf_upload"
    upload_dir.mkdir()
    staged = upload_dir / "README.md"
    staged.write_text("# Custom README", encoding="utf-8")
    mocker.patch("aiaccel.torch.apps.upload_hf.ask_confirm", return_value=False)

    assert upload_hf.stage_readme(tmp_path, upload_dir, "test/model")
    assert staged.read_text(encoding="utf-8") == "# Custom README"


def test_upload_model_aborts_when_not_confirmed(mocker: MockerFixture, tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# model", encoding="utf-8")
    mocker.patch("aiaccel.torch.apps.upload_hf.ask_confirm", return_value=False)
    mocker.patch("aiaccel.torch.apps.upload_hf.validate_readme", return_value=True)
    login = mocker.patch("aiaccel.torch.apps.upload_hf.login")
    api = mocker.patch("aiaccel.torch.apps.upload_hf.HfApi")

    upload_hf.upload_model(tmp_path, "test/model")

    login.assert_not_called()
    api.assert_not_called()


def test_upload_model_reconfirms_changed_files(mocker: MockerFixture, tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("# model", encoding="utf-8")
    mocker.patch("aiaccel.torch.apps.upload_hf.ask_confirm", side_effect=[True, False])
    mocker.patch("aiaccel.torch.apps.upload_hf.login")
    mocker.patch("aiaccel.torch.apps.upload_hf.validate_readme", return_value=True)
    api = mocker.Mock()
    api.repo_exists.return_value = True
    mocker.patch("aiaccel.torch.apps.upload_hf.HfApi", return_value=api)
    manifest = upload_hf.get_file_manifest(tmp_path)
    mocker.patch(
        "aiaccel.torch.apps.upload_hf.get_file_manifest",
        side_effect=[manifest, {**manifest, "unexpected.ckpt": (12, 1)}],
    )

    upload_hf.upload_model(tmp_path, "test/model")

    api.upload_folder.assert_not_called()


def test_select_state_dict_prefixes(mocker: MockerFixture) -> None:
    state_dict = {
        "model.encoder.layers.0.weight": torch.tensor([1.0]),
        "model.encoder.layers.1.weight": torch.tensor([2.0]),
        "model.decoder.weight": torch.tensor([3.0]),
    }
    checkbox = mocker.patch("aiaccel.torch.apps.upload_hf.questionary.checkbox")
    checkbox.return_value.unsafe_ask.return_value = ["model.encoder.layers"]

    selected = upload_hf.select_state_dict_prefixes(state_dict)

    assert selected == ["model.encoder.layers"]
    choices = checkbox.call_args.kwargs["choices"]
    assert [(choice.value, choice.title) for choice in choices] == [
        ("model.decoder.weight", "model.decoder.weight (1 keys)"),
        ("model.encoder.layers", "model.encoder.layers (2 keys)"),
    ]


def test_prepare_checkpoint_repeats_prefix_selection(mocker: MockerFixture) -> None:
    checkpoint = {
        "epoch": 1,
        "pytorch-lightning_version": "2.5.0",
        "hyper_parameters": {"lr": 0.001},
        "hparams_name": "kwargs",
        "hparams_type": "dict",
        "optimizer_states": [{"step": 10}],
        "state_dict": {
            "model.encoder.layer.0": torch.tensor([1.0]),
            "model.encoder.layer.1": torch.tensor([2.0]),
            "model.encoder_extra.weight": torch.tensor([3.0]),
            "loss.weight": torch.tensor([4.0]),
        },
    }
    mocker.patch("aiaccel.torch.apps.upload_hf.ask_confirm", side_effect=[True, True, True])
    select = mocker.patch(
        "aiaccel.torch.apps.upload_hf.select_state_dict_prefixes",
        side_effect=[["model.encoder"], ["loss"], []],
    )

    cleaned = upload_hf.prepare_checkpoint(checkpoint)

    assert set(cleaned) == {
        "state_dict",
        "pytorch-lightning_version",
        "hyper_parameters",
        "hparams_name",
        "hparams_type",
    }
    assert cleaned["pytorch-lightning_version"] == "2.5.0"
    assert cleaned["hyper_parameters"] == {"lr": 0.001}
    assert cleaned["hparams_name"] == "kwargs"
    assert cleaned["hparams_type"] == "dict"
    assert list(cleaned["state_dict"]) == ["model.encoder_extra.weight"]
    assert len(checkpoint["state_dict"]) == 4
    assert select.call_count == 3


def test_prepare_checkpoint_omits_missing_optional_metadata(mocker: MockerFixture) -> None:
    checkpoint = {"epoch": 1, "state_dict": {"weight": torch.tensor([1.0])}}
    mocker.patch("aiaccel.torch.apps.upload_hf.ask_confirm", return_value=True)
    mocker.patch("aiaccel.torch.apps.upload_hf.select_state_dict_prefixes", return_value=[])

    cleaned = upload_hf.prepare_checkpoint(checkpoint)

    assert list(cleaned) == ["state_dict"]


def test_prepare_checkpoint_keeps_metadata_by_default(mocker: MockerFixture) -> None:
    checkpoint = {"epoch": 1, "state_dict": {"weight": torch.tensor([1.0])}}
    mocker.patch("aiaccel.torch.apps.upload_hf.ask_confirm", return_value=False)
    assert upload_hf.prepare_checkpoint(checkpoint) is checkpoint


def test_ask_languages_retries_invalid_code(mocker: MockerFixture) -> None:
    mocker.patch("aiaccel.torch.apps.upload_hf.ask_text", side_effect=["English", "en, ja"])

    assert upload_hf.ask_languages() == ["en", "ja"]


def test_upload_model_rejects_invalid_readme_before_login(mocker: MockerFixture, tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text(
        "---\nlanguage:\n- English\npipeline_tag: ov-sed\n---\n", encoding="utf-8"
    )
    validate = mocker.patch("aiaccel.torch.apps.upload_hf.ModelCard.load")
    validate.return_value.validate.side_effect = ValueError("Invalid metadata")
    login = mocker.patch("aiaccel.torch.apps.upload_hf.login")

    upload_hf.upload_model(tmp_path, "test/model")

    login.assert_not_called()
    validate.return_value.validate.assert_called_once_with(repo_type="model")
