from typing import Union
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pathlib import Path

from importlib.resources import read_text


def load_config(config_path: str) -> Union[None, DictConfig]:
    """
    Load any configuration files, return the DictConfig object.
    Args:
        config_path (str): A path to a configuration file.

    Returns:
        config: DictConfig object
    """
    path = Path(config_path).resolve()

    if not path.exists():
        return None
    default = OmegaConf.create(read_text('aiaccel', 'default_config.yaml'))
    customize = OmegaConf.load(path)

    config = OmegaConf.merge(default, customize)
    config.config_path = str(path)

    return config
