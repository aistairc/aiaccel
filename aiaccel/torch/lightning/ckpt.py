from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
from omegaconf import OmegaConf as oc  # noqa: N813

from torch import nn


def from_pretrained(
    model_path: str | Path,
    config_name: str = "merged_config.yaml",
    overwrite_config: DictConfig | ListConfig | dict | list | None = None,
    device: str = "cuda",
    eval_mode: bool = True,
) -> tuple[nn.Module, DictConfig | ListConfig]:
    """
    Load a model from a checkpoint.

    Args:
        model_path (str | Path): Path to the model directory.
        config_name (str): Name of the configuration file. Default is "merged_config.yaml".
        overwrite_config (DictConfig | ListConfig | dict | list | None): Configuration to overwrite the
            loaded configuration. Default is None.
        device (str): Device to load the model onto. Default is "cuda".

    Returns:
        model (nn.Module): The loaded model.
        config (DictConfig | ListConfig): The loaded configuration.
    """
    model_path = Path(model_path)

    config = oc.load(model_path / config_name)

    if overwrite_config is not None:
        config = oc.merge(config, overwrite_config)

    checkpoint_filename = config.checkpoint_filename if "checkpoint_filename" in config else "last.ckpt"
    checkpoint_path = model_path / "checkpoints" / checkpoint_filename
    config.task._target_ += ".load_from_checkpoint"
    model = instantiate(
        config.task,
        checkpoint_path=checkpoint_path,
        map_location=device,
    )

    if eval_mode:
        model.eval()

    return model, config
