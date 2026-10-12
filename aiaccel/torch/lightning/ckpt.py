# Copyright (C) 2025 National Institute of Advanced Industrial Science and Technology (AIST)
# SPDX-License-Identifier: MIT

from typing import Any

import logging
from pathlib import Path
import warnings

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as oc  # noqa: N813

from torch import nn

from huggingface_hub import snapshot_download

from aiaccel.config import prepare_config

logger = logging.getLogger(__name__)


def get_checkpoint_path(model_path: str | Path, checkpoint_filename: str) -> Path:
    if not checkpoint_filename.endswith(".ckpt"):
        checkpoint_filename += ".ckpt"

    return Path(model_path) / "checkpoints" / checkpoint_filename


def load_checkpoint(
    model_path: str | Path,
    config_name: str = "merged_config.yaml",
    device: str = "cuda",
    overwrite_config: DictConfig | ListConfig | dict[Any, Any] | list[Any] | None = None,
    strict: bool | None = None,
    weights_only: bool | None = None,
) -> tuple[nn.Module, DictConfig | ListConfig]:
    """
    Load a PyTorch Lightning model from a pre-trained checkpoint.

    This function loads a model from a specified path, which can be a local directory
    or a Hugging Face repository. It also loads the associated configuration file and
    allows for optional configuration overrides. The model can be set to evaluation mode
    if specified.

    Args:
        model_path (str | Path): The path to the model directory or Hugging Face repo.
            For local paths, use the format "file://<absolute_path>" or just the path (str | Path).
            For Hugging Face, use the format "hf://<repo_id>".
        config_name (str): The name of the configuration file to load. Default is "merged_config.yaml".
        device (str): The device to map the model to. Default is "cuda".
        overwrite_config (DictConfig | ListConfig | dict | list | None): Optional configuration overrides.
        strict (bool | None): Override task.strict; None keeps the existing configuration or Lightning default.
        weights_only (bool | None): Override task.weights_only; None preserves the existing setting.
    """

    if isinstance(model_path, str):
        if model_path.startswith("hf://"):
            logger.info("Downloading model from Hugging Face...")
            hf_path = model_path.removeprefix("hf://")
            model_path = Path(snapshot_download(hf_path))
        elif model_path.startswith("file://"):
            model_path = Path(model_path.removeprefix("file://"))
        else:
            model_path = Path(model_path)

    config_path = model_path / config_name
    task_options = {}
    if strict is not None:
        task_options["strict"] = strict
    if weights_only is not None:
        task_options["weights_only"] = weights_only
    if task_options:
        overwrite_config = oc.merge(overwrite_config if overwrite_config is not None else {}, {"task": task_options})
    config = prepare_config(config_path, overwrite_config=overwrite_config)
    checkpoint_path = get_checkpoint_path(model_path, config.checkpoint_filename)

    logger.info(f"Loading model from {checkpoint_path}...")

    config.task._target_ += ".load_from_checkpoint"
    effective_strict = config.task.get("strict")
    if effective_strict is not None:
        logger.warning("Checkpoint loading with strict=%s.", effective_strict)

    with warnings.catch_warnings():
        if effective_strict is False:
            warnings.filterwarnings("ignore", message="Found keys that are", category=UserWarning)
        model = instantiate(
            config.task,
            checkpoint_path=checkpoint_path,
            map_location=device,
        )

    return model, config
