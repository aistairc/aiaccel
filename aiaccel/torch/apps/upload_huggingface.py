# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing import Any

from argparse import ArgumentParser
from pathlib import Path

from omegaconf import OmegaConf as oc  # noqa: N813

import torch

from huggingface_hub import create_repo, login, repo_exists, upload_file

from aiaccel.config import print_config


def remove_fullpath(obj: dict[str, Any] | list[Any] | Any) -> Any:
    """Recursively remove elements containing full paths from a dict.

    Args:
        obj (dict[str, Any] | list[Any] | Any): The object that contains a dict from which full paths should be removed.

    Returns:
        Any: The object with full paths removed.
    """
    if isinstance(obj, dict):
        return {
            key: remove_fullpath(value)
            for key, value in obj.items()
            if not (isinstance(value, str) and Path(value).is_absolute())
        }
    elif isinstance(obj, list):
        return [remove_fullpath(v) for v in obj if not (isinstance(v, str) and Path(v).is_absolute())]
    else:
        return obj


def yes_no_input() -> bool:
    while True:
        choice = input("Please respond with 'yes' or 'no' [y/N]: ").lower()
        if choice in ["y", "ye", "yes"]:
            return True
        elif choice in ["n", "no"]:
            return False


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--config_path", type=Path)
    parser.add_argument("--save_config_filename", type=str, default="pathremoved_config.yaml")
    parser.add_argument("--ckpt_path", type=Path)
    parser.add_argument("--save_ckpt_filename", type=str, default="pathremoved.ckpt")
    parser.add_argument("--repo_id", type=str)
    parser.add_argument("--repo_type", type=str)

    args = parser.parse_args()

    config = oc.to_container(oc.load(args.config_path))
    config = remove_fullpath(config)
    config["checkpoint_filename"] = args.save_ckpt_filename

    with open(args.config_path.parent / args.save_config_filename, "w") as f:
        oc.save(config, f)

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    ckpt = remove_fullpath(ckpt)

    torch.save(ckpt, args.ckpt_path.parent / args.save_ckpt_filename)

    if args.repo_id and args.repo_type:
        print_config(config)
        print("The above configuration file and model checkpoint will be uploaded to Hugging Face. Is that OK?")
        if yes_no_input():
            # Upload config and ckpt in Hugging Face
            login()

            if not repo_exists(repo_id=args.repo_id, repo_type=args.repo_type):
                print(f"The repository {args.repo_id} was not found. Would you like to create a new one?")
                if yes_no_input():
                    # Create repository
                    create_repo(repo_id=args.repo_id, repo_type=args.repo_type)
                else:
                    return

            upload_file(
                path_or_fileobj=args.config_path.parent / args.save_config_filename,
                path_in_repo=args.save_config_filename,
                repo_id=args.repo_id,
                repo_type=args.repo_type,
            )
            upload_file(
                path_or_fileobj=args.ckpt_path.parent / args.save_ckpt_filename,
                path_in_repo="checkpoints/" + args.save_ckpt_filename,
                repo_id=args.repo_id,
                repo_type=args.repo_type,
            )


if __name__ == "__main__":
    main()
