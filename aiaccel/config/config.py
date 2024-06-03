from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import read_text
from pathlib import Path

from omegaconf import OmegaConf
from omegaconf.base import Container
from omegaconf.dictconfig import DictConfig


@dataclass
class Config:
    timeout: int
    n_max_jobs: int
    sampler: str
    group: str
    direction: str
    n_trials: int
    seed: int | None


def set_structs_false(conf: Container) -> None:
    OmegaConf.set_struct(conf, False)
    if hasattr(conf, "__iter__"):
        for item in conf:
            if isinstance(conf.__dict__["_content"], dict):
                set_structs_false(conf.__dict__["_content"][item])


def load_config(config_path: Path | str | None) -> DictConfig:
    base = OmegaConf.structured(Config)
    default = OmegaConf.create(read_text("aiaccel.config", "default.yaml"))
    config = OmegaConf.merge(base, default)
    set_structs_false(config)
    if config_path is not None:
        path = Path(config_path).resolve()
        if not path.exists():
            raise ValueError("Config is not found.")
        customize = OmegaConf.load(path)
        config = OmegaConf.merge(config, customize)

    if not isinstance(config, DictConfig):
        raise RuntimeError("The configuration is not a DictConfig object.")

    return config
