from typing import Any

import logging
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as oc  # noqa: N813

from torch import nn

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


def load_checkpoint(
    model_path: str | Path,
    config_name: str = "merged_config.yaml",
    device: str = "cuda",
    overwrite_config: DictConfig | ListConfig | dict[Any, Any] | list[Any] | None = None,
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

    config = oc.load(model_path / config_name)

    if overwrite_config is not None:
        config = oc.merge(config, overwrite_config)

    checkpoint_filename = config.checkpoint_filename if "checkpoint_filename" in config else "last.ckpt"
    checkpoint_path = model_path / "checkpoints" / checkpoint_filename

    logger.info(f"Loading model from {checkpoint_path}...")

    config.task._target_ += ".load_from_checkpoint"
    model = instantiate(
        config.task,
        checkpoint_path=checkpoint_path,
        map_location=device,
    )

    return model, config
