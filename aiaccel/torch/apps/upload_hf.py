# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT


from argparse import ArgumentParser
from collections.abc import Callable
import json
from pathlib import Path
import shutil

from omegaconf import DictConfig
from omegaconf import OmegaConf as oc  # noqa: N813
from rich.console import Console
from rich.prompt import Confirm, Prompt

import torch

from huggingface_hub import HfApi, login

from aiaccel.torch.utils.remove_absolute_path import remove_absolute_path

console = Console()


def wait_for_review(path: Path) -> None:
    console.print(f"\nPlease review:\n  [bold]{path}[/bold]")
    console.input("Press Enter when ready to continue...")


def resolve_checkpoint(model_dir: Path, config: DictConfig) -> Path:
    checkpoint_filename = config.get("checkpoint_filename")
    if not isinstance(checkpoint_filename, str) or not checkpoint_filename:
        raise ValueError("merged_config.yaml must contain a non-empty 'checkpoint_filename'.")

    if not checkpoint_filename.endswith(".ckpt"):
        checkpoint_filename += ".ckpt"

    checkpoint_path = model_dir / "checkpoints" / checkpoint_filename
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    return checkpoint_path


def create_model_card(model_dir: Path, repo_id: str) -> Path:
    readme_path = model_dir / "README.md"

    model_name = Prompt.ask("Model name", default=repo_id.rsplit("/", 1)[-1])
    license_name = Prompt.ask(
        "License",
        choices=["apache-2.0", "mit", "cc-by-4.0", "other", "skip"],
        default="apache-2.0",
    )
    pipeline_tag = Prompt.ask("Task / pipeline tag", default="")
    language_input = Prompt.ask("Language(s), comma-separated", default="")
    description = Prompt.ask("Short description", default="")

    metadata = ["---"]
    if license_name != "skip":
        metadata.append(f"license: {json.dumps(license_name)}")
    if pipeline_tag:
        metadata.append(f"pipeline_tag: {json.dumps(pipeline_tag)}")
    metadata.append('library_name: "aiaccel"')

    languages = [language.strip() for language in language_input.split(",") if language.strip()]
    if languages:
        metadata.append("language:")
        metadata.extend(f"  - {json.dumps(language)}" for language in languages)

    metadata.append("---")

    readme = "\n".join(
        [
            *metadata,
            "",
            f"# {model_name}",
            "",
            description or "TODO: Describe the model.",
            "",
            "## Usage",
            "",
            "TODO: Add usage instructions.",
            "",
            "## Training",
            "",
            "TODO: Describe the training procedure.",
            "",
            "## Limitations",
            "",
            "TODO: Describe known limitations.",
            "",
        ]
    )

    readme_path.write_text(readme, encoding="utf-8")
    console.print(f"Created model card: [bold]{readme_path}[/bold]")
    return readme_path


def print_cleaning_report(label: str, removed: list[str]) -> None:
    if not removed:
        console.print(f"No absolute paths were removed from {label}.")
        return

    console.print(f"Removed entries from [bold]{label}[/bold]:")
    for path in removed:
        console.print(f"  - {path}")


def stage_file(target: Path, label: str, generate: Callable[[], None]) -> bool:
    if target.is_file():
        console.print(f"\n[bold]{label}[/bold] already exists:\n  {target}")
        wait_for_review(target)
        if Confirm.ask(f"Use this {label} for upload?", default=True):
            return True

    generate()

    wait_for_review(target)
    return Confirm.ask(f"Use this {label} for upload?", default=True)


def stage_readme(model_dir: Path, upload_dir: Path, repo_id: str) -> bool:
    source = model_dir / "README.md"
    target = upload_dir / "README.md"

    if target.is_file():
        console.print(f"\n[bold]README.md[/bold] already exists:\n  {target}")
        wait_for_review(target)
        if Confirm.ask("Use this README.md for upload?", default=True):
            return True

    if not source.is_file():
        console.print("README.md was not found.")
        if not Confirm.ask("Create a model card?", default=True):
            return False
        source = create_model_card(model_dir, repo_id)

    wait_for_review(source)
    if not Confirm.ask("Use this README.md for upload?", default=True):
        return False

    shutil.copy2(source, target)
    return True


def stage_config(config_path: Path, upload_dir: Path) -> bool:
    target = upload_dir / "merged_config.yaml"

    def generate() -> None:
        if not Confirm.ask("Clean absolute paths from merged_config.yaml?", default=True):
            shutil.copy2(config_path, target)
            return

        config = oc.to_container(oc.load(config_path), resolve=False)
        cleaned, removed = remove_absolute_path(config)
        oc.save(cleaned, target)
        print_cleaning_report("merged_config.yaml", removed)

    return stage_file(target, "merged_config.yaml", generate)


def stage_checkpoint(checkpoint_path: Path, upload_dir: Path) -> bool:
    target = upload_dir / "checkpoints" / checkpoint_path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    label = target.relative_to(upload_dir).as_posix()

    def generate() -> None:
        if not Confirm.ask(f"Clean absolute paths from {checkpoint_path.name}?", default=True):
            shutil.copy2(checkpoint_path, target)
            return

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cleaned, removed = remove_absolute_path(checkpoint)
        torch.save(cleaned, target)
        print_cleaning_report(checkpoint_path.name, removed)

    return stage_file(target, label, generate)


def upload_model(upload_dir: Path, repo_id: str) -> None:
    console.print("\n[bold]Files to upload:[/bold]")
    for path in sorted(upload_dir.rglob("*")):
        if path.is_file():
            console.print(f"  {path.relative_to(upload_dir).as_posix()}")

    console.print(f"\nRepository:\n  [bold]{repo_id}[/bold]")

    if not Confirm.ask("Upload these files?", default=True):
        return

    login()
    api = HfApi()

    if not api.repo_exists(repo_id=repo_id, repo_type="model"):
        if not Confirm.ask(f"Repository {repo_id} does not exist. Create it?", default=True):
            return
        api.create_repo(repo_id=repo_id, repo_type="model")

    api.upload_folder(
        folder_path=upload_dir,
        repo_id=repo_id,
        repo_type="model",
    )
    console.print(f"Uploaded to https://huggingface.co/{repo_id}")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("repo_id")
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    config_path = model_dir / "merged_config.yaml"

    if not config_path.is_file():
        parser.error(f"Config not found: {config_path}")

    config = oc.load(config_path)
    if not isinstance(config, DictConfig):
        parser.error(f"Expected a mapping in {config_path}")

    try:
        checkpoint_path = resolve_checkpoint(model_dir, config)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))

    console.print(f"Config:\n  {config_path}")
    console.print(f"Checkpoint:\n  {checkpoint_path}")

    upload_dir = model_dir / "hf_upload"
    upload_dir.mkdir(parents=True, exist_ok=True)

    if not stage_readme(model_dir, upload_dir, args.repo_id):
        return
    if not stage_config(config_path, upload_dir):
        return
    if not stage_checkpoint(checkpoint_path, upload_dir):
        return

    upload_model(upload_dir, args.repo_id)


if __name__ == "__main__":
    main()
