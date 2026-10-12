# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing import Any

from argparse import ArgumentParser
from collections import Counter
from collections.abc import Callable
from pathlib import Path
import re
import shutil

from omegaconf import DictConfig
from omegaconf import OmegaConf as oc  # noqa: N813
from rich.console import Console

import torch

import questionary
from huggingface_hub import HfApi, ModelCard, ModelCardData, login

from aiaccel.torch.lightning.ckpt import get_checkpoint_path
from aiaccel.torch.utils.remove_absolute_path import remove_absolute_path


rprint = Console(highlight=False).print


def ask_confirm(message: str, default: bool = False) -> bool:
    return questionary.confirm(message, default=default).unsafe_ask()


def ask_text(message: str, default: str = "") -> str:
    return questionary.text(message, default=default).unsafe_ask()


PIPELINE_TAGS = (
    "audio-classification", "audio-to-audio", "automatic-speech-recognition",
    "depth-estimation", "feature-extraction", "fill-mask", "image-classification",
    "image-feature-extraction", "image-segmentation", "image-text-to-image",
    "image-text-to-text", "image-text-to-video", "image-to-3d", "image-to-image",
    "image-to-text", "image-to-video", "keypoint-detection", "mask-generation",
    "object-detection", "other", "reinforcement-learning", "robotics",
    "sentence-similarity", "summarization", "table-question-answering",
    "tabular-classification", "tabular-regression", "text-classification",
    "text-generation", "text-ranking", "text-retrieval", "text-to-audio",
    "text-to-image", "text-to-speech", "text-to-video", "time-series-forecasting",
    "token-classification", "translation", "video-classification",
    "video-text-to-text", "video-to-video", "visual-question-answering",
    "voice-activity-detection", "zero-shot-classification",
    "zero-shot-image-classification", "zero-shot-object-detection",
)


def ask_languages() -> list[str] | None:
    """Accept standard language codes, not display names such as English."""
    while True:
        raw = ask_text("Language code(s), comma-separated (e.g. en, ja; blank to skip)")
        codes = [value.strip() for value in raw.split(",") if value.strip()]
        if not codes:
            return None
        if all(re.fullmatch(r"[a-z]{2,3}|code|multilingual", code) for code in codes):
            return codes
        rprint("Use lowercase ISO language codes such as en or ja (not English or Japanese).")


def validate_readme(upload_dir: Path) -> bool:
    """Validate staged model-card metadata before any Hugging Face upload."""
    path = upload_dir / "README.md"
    if not path.is_file():
        rprint(f"Model README.md not found: {path}")
        return False
    try:
        ModelCard.load(str(path)).validate(repo_type="model")
    except ValueError as exc:
        rprint(f"[bold]Invalid README.md metadata in {path}:[/bold]\n{exc}")
        rprint("Edit the staged README.md and run the command again.")
        return False
    return True


def should_overwrite(target: Path) -> bool:
    """Prompt only when a staged file already exists."""
    return not target.exists() or ask_confirm(f"Overwrite {target.name}?", default=False)


def prepare_upload_dir(upload_dir: Path) -> None:
    """Optionally clear the previous staging directory."""
    if upload_dir.exists() and ask_confirm("Delete existing hf_upload?", default=False):
        shutil.rmtree(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)


def stage_readme(model_dir: Path, upload_dir: Path, repo_id: str) -> bool:
    """Prepare README.md for the upload directory."""
    target = upload_dir / "README.md"

    if not should_overwrite(target):
        return True

    source = model_dir / "README.md"

    # Recreate the source model card only when requested.
    if source.is_file():
        create_model_card = ask_confirm("Regenerate model README.md?", default=False)
    else:
        rprint("README.md was not found.")
        create_model_card = ask_confirm("Create a model card?", default=True)
        if not create_model_card:
            return False

    if create_model_card:
        model_name = ask_text("Model name", default=repo_id.rsplit("/", 1)[-1])
        license_name = ask_text(
            "License (e.g. apache-2.0, mit, cc-by-4.0; skip to omit)",
            default="apache-2.0",
        )
        library_name = ask_text("Library", default="aiaccel")
        pipeline_tag = questionary.select(
            "Task / pipeline tag (choose 'skip' to omit)",
            choices=["skip", *PIPELINE_TAGS],
        ).unsafe_ask()
        languages = ask_languages()
        description = ask_text("Short description", default="")

        card_data = ModelCardData(
            model_name=model_name,
            license=None if license_name == "skip" else license_name,
            pipeline_tag=None if pipeline_tag == "skip" else pipeline_tag,
            language=languages,
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

    shutil.copy2(source, target)

    return True


def stage_cleaned_file(
    source: Path,
    target: Path,
    load: Callable[[Path], Any],
    save: Callable[[Any, Path], None],
) -> None:
    """Prepare a config or checkpoint, removing absolute paths when written."""
    if not should_overwrite(target):
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    cleaned, removed = remove_absolute_path(load(source))
    save(cleaned, target)

    if removed:
        rprint(f"Removed entries from [bold]{source.name}[/bold]:")
        for entry in removed:
            rprint(f"  - {entry}")


def select_state_dict_prefixes(state_dict: dict[str, Any]) -> list[str]:
    """Present up to three key segments as selectable prefixes."""
    prefixes = Counter(".".join(key.split(".")[:3]) for key in state_dict)
    choices = [
        questionary.Choice(f"{prefix} ({count} keys)", value=prefix)
        for prefix, count in sorted(prefixes.items())
    ]
    return questionary.checkbox(
        "Select state_dict prefixes to remove (Space: select, Enter: finish)",
        choices=choices,
    ).unsafe_ask()


def prepare_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Optionally keep only state_dict and remove selected key prefixes."""
    if not ask_confirm("Keep only state_dict?", default=False):
        return checkpoint

    if not isinstance(checkpoint.get("state_dict"), dict):
        raise ValueError("Checkpoint has no state_dict mapping.")

    state_dict = checkpoint["state_dict"].copy()
    while state_dict:
        selected = select_state_dict_prefixes(state_dict)
        if not selected:
            break

        matches = [
            key
            for key in state_dict
            if any(key == prefix or key.startswith(f"{prefix}.") for prefix in selected)
        ]
        if ask_confirm(f"Remove {len(matches)} selected keys?", default=False):
            for key in matches:
                del state_dict[key]

    return {"state_dict": state_dict}


def get_file_manifest(upload_dir: Path) -> dict[str, tuple[int, int]]:
    """Snapshot filenames, sizes, and modification times for upload review."""
    manifest: dict[str, tuple[int, int]] = {}
    for path in sorted(upload_dir.rglob("*")):
        if path.is_file():
            stat = path.stat()
            manifest[path.relative_to(upload_dir).as_posix()] = (stat.st_size, stat.st_mtime_ns)
    return manifest


def show_upload_files(manifest: dict[str, tuple[int, int]]) -> None:
    for path, (size, _) in manifest.items():
        rprint(f"  {path} ({size:,} bytes)")


def upload_model(upload_dir: Path, repo_id: str) -> None:
    """Review and upload the staged model directory to Hugging Face Hub."""
    manifest = get_file_manifest(upload_dir)
    if not manifest:
        rprint("No files to upload.")
        return
    if not validate_readme(upload_dir):
        return

    rprint(f"\nRepository: [bold]{repo_id}[/bold]")
    rprint("\n[bold]Files to upload:[/bold]")
    show_upload_files(manifest)
    if not ask_confirm("Upload all these files?", default=False):
        return

    login()
    api = HfApi()

    if not api.repo_exists(repo_id=repo_id, repo_type="model"):
        if not ask_confirm(f"Repository {repo_id} does not exist. Create it as private?", default=True):
            return
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=True,
        )

    # Review the current contents again if files changed during authentication.
    updated_manifest = get_file_manifest(upload_dir)
    if updated_manifest != manifest:
        rprint("\n[bold]Files changed since confirmation:[/bold]")
        show_upload_files(updated_manifest)
        if not updated_manifest or not ask_confirm("Upload these files instead?", default=False):
            return

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

    # Prepare and stage the files before the single upload review.
    upload_dir = model_dir / "hf_upload"
    prepare_upload_dir(upload_dir)
    if not stage_readme(model_dir, upload_dir, args.repo_id):
        return

    stage_cleaned_file(
        config_path,
        upload_dir / "merged_config.yaml",
        lambda path: oc.to_container(oc.load(path), resolve=False),
        oc.save,
    )
    stage_cleaned_file(
        checkpoint_path,
        upload_dir / "checkpoints" / checkpoint_path.name,
        lambda path: prepare_checkpoint(torch.load(path, map_location="cpu", weights_only=False)),
        torch.save,
    )
    upload_model(upload_dir, args.repo_id)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        rprint("\nCancelled.")
