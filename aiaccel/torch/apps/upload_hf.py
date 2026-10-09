# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing import Any

from argparse import ArgumentParser
from collections.abc import Callable
from pathlib import Path
import shutil

from omegaconf import DictConfig
from omegaconf import OmegaConf as oc  # noqa: N813
from rich import print as rprint
from rich.prompt import Confirm, Prompt

import torch

from huggingface_hub import HfApi, ModelCard, ModelCardData, login

from aiaccel.torch.lightning.ckpt import get_checkpoint_path
from aiaccel.torch.utils.remove_absolute_path import remove_absolute_path


def _review_file(path: Path) -> bool:
    """Pause for manual review and confirm whether the file should be uploaded."""
    rprint(f"\nPlease review:\n  [bold]{path}[/bold]")
    input("Press Enter when ready to continue...")
    return Confirm.ask(f"Use {path.name} for upload?", default=True)


def stage_readme(model_dir: Path, upload_dir: Path, repo_id: str) -> bool:
    """Prepare README.md for the upload directory."""
    target = upload_dir / "README.md"

    # Prefer an existing staged README when available.
    if target.is_file() and _review_file(target):
        return True

    source = model_dir / "README.md"

    # Create a model card in the original model directory when README.md is missing.
    if not source.is_file():
        rprint("README.md was not found.")
        if not Confirm.ask("Create a model card?", default=True):
            return False

        model_name = Prompt.ask("Model name", default=repo_id.rsplit("/", 1)[-1])
        license_name = Prompt.ask(
            "License",
            choices=["apache-2.0", "mit", "cc-by-4.0", "other", "skip"],
            default="apache-2.0",
        )
        library_name = Prompt.ask("Library", default="aiaccel")
        pipeline_tag = Prompt.ask("Task / pipeline tag", default="")
        language = Prompt.ask("Language(s), comma-separated", default="")
        description = Prompt.ask("Short description", default="")

        card_data = ModelCardData(
            model_name=model_name,
            license=None if license_name == "skip" else license_name,
            pipeline_tag=pipeline_tag or None,
            language=[value.strip() for value in language.split(",") if value.strip()] or None,
            library_name=library_name,
        )
        template_path = Path(__file__).parent / "templates" / "model_card.md"

        card = ModelCard.from_template(
            card_data=card_data,
            template_path=str(template_path),
            model_name=model_name,
            description=description,
        )
        card.save(source)

        rprint(f"Created model card: [bold]{source}[/bold]")

    # Review the source README before copying it into the upload directory.
    if not _review_file(source):
        return False

    shutil.copy2(source, target)
    return True


def stage_cleaned_file(
    source: Path,
    target: Path,
    load: Callable[[Path], Any],
    save: Callable[[Any, Path], None],
) -> bool:
    """Prepare a config or checkpoint, optionally removing absolute paths."""
    # Reuse an existing staged file if the user approves it.
    if target.is_file() and _review_file(target):
        return True

    target.parent.mkdir(parents=True, exist_ok=True)

    # Otherwise create a new staged file, optionally removing absolute paths.
    if Confirm.ask(f"Clean absolute paths from {source.name}?", default=True):
        cleaned, removed = remove_absolute_path(load(source))
        save(cleaned, target)

        if removed:
            rprint(f"Removed entries from [bold]{source.name}[/bold]:")
            for entry in removed:
                rprint(f"  - {entry}")
        else:
            rprint(f"No absolute paths were removed from {source.name}.")
    else:
        shutil.copy2(source, target)

    return _review_file(target)


def upload_model(upload_dir: Path, repo_id: str) -> None:
    """Upload the staged model directory to Hugging Face Hub."""
    # Show the exact set of files that will be uploaded.
    rprint("\n[bold]Files to upload:[/bold]")
    for path in sorted(upload_dir.rglob("*")):
        if path.is_file():
            rprint(f"  {path.relative_to(upload_dir).as_posix()}")

    rprint(f"\nRepository:\n  [bold]{repo_id}[/bold]")
    if not Confirm.ask("Upload these files?", default=True):
        return

    # Authenticate and create the model repository when necessary.
    login()
    api = HfApi()

    if not api.repo_exists(repo_id=repo_id, repo_type="model"):
        if not Confirm.ask(f"Repository {repo_id} does not exist. Create it as private?", default=True):
            return
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=True,
        )

    # Upload only the reviewed contents of hf_upload/.
    api.upload_folder(
        folder_path=upload_dir,
        repo_id=repo_id,
        repo_type="model",
    )
    rprint(f"Uploaded to https://huggingface.co/{repo_id}")


def main() -> None:
    # Parse the local model directory and Hugging Face repository ID.
    parser = ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("repo_id")
    args = parser.parse_args()

    # Load the training config and determine the checkpoint to upload.
    model_dir = args.model_dir.resolve()
    config_path = model_dir / "merged_config.yaml"
    if not config_path.is_file():
        parser.error(f"Config not found: {config_path}")

    config = oc.load(config_path)
    if not isinstance(config, DictConfig):
        parser.error(f"Expected a mapping in {config_path}")

    checkpoint_filename = config.get("checkpoint_filename")
    if not isinstance(checkpoint_filename, str) or not checkpoint_filename:
        parser.error("merged_config.yaml must contain a non-empty 'checkpoint_filename'.")

    checkpoint_path = get_checkpoint_path(model_dir, checkpoint_filename)
    if not checkpoint_path.is_file():
        parser.error(f"Checkpoint not found: {checkpoint_path}")

    rprint(f"Config:\n  {config_path}")
    rprint(f"Checkpoint:\n  {checkpoint_path}")

    # Prepare the directory that mirrors the files uploaded to Hugging Face.
    upload_dir = model_dir / "hf_upload"
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Prepare and review each upload artifact.
    if not stage_readme(model_dir, upload_dir, args.repo_id):
        return

    if not stage_cleaned_file(
        config_path,
        upload_dir / "merged_config.yaml",
        lambda path: oc.to_container(oc.load(path), resolve=False),
        oc.save,
    ):
        return

    if not stage_cleaned_file(
        checkpoint_path,
        upload_dir / "checkpoints" / checkpoint_path.name,
        lambda path: torch.load(path, map_location="cpu", weights_only=False),
        torch.save,
    ):
        return

    upload_model(upload_dir, args.repo_id)


if __name__ == "__main__":
    main()
