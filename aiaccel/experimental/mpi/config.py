from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from importlib.resources import read_text
from pathlib import Path
from typing import Any, List, Optional, Union

from omegaconf import OmegaConf
from omegaconf.base import Container
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig


class ResourceType(Enum):
    abci: str = "abci"
    local: str = "local"
    python_local: str = "python_local"
    mpi: str = "mpi"

    @classmethod
    def _missing_(cls, value: Any) -> Any | None:
        value = value.lower()
        for member in cls:
            if member.value == value:
                return member
        return None


class OptimizerDirection(Enum):
    minimize: str = "minimize"
    maximize: str = "maximize"

    @classmethod
    def _missing_(cls, value: Any) -> Any | None:
        value = value.lower()
        for member in cls:
            if member.value == value:
                return member
        return None


@dataclass
class GenericConfig:
    workspace: str
    job_command: str
    python_file: str
    function: str
    batch_job_timeout: int
    sleep_time: Union[float, int]
    is_ignore_warning: bool


@dataclass
class ResourceConifig:
    type: ResourceType
    num_node: int
    mpi_npernode: int
    mpi_enviroment: str
    mpi_bat_rt_type: str
    mpi_bat_rt_num: int
    mpi_bat_h_rt: str
    mpi_bat_root_dir: str
    mpi_bat_venv_dir: str
    mpi_bat_aiaccel_dir: str
    mpi_bat_config_dir: str
    mpi_bat_file: str
    mpi_hostfile: str
    mpi_gpu_mode: bool
    mpi_bat_make_file: bool


@dataclass
class AbciConifig:
    group: str
    job_script_preamble: str
    job_execution_options: Optional[str]
    runner_search_pattern: Optional[str]


@dataclass
class ParameterConfig:
    name: str
    type: str
    lower: Union[float, int]
    upper: Union[float, int]
    # initial: Optional[Union[None, float, int, str, List[Union[float, int]]]]  # Unions of containers are not supported
    initial: Optional[Any]
    choices: Optional[List[Union[None, float, int, str]]]
    sequence: Optional[List[Union[None, float, int, str]]]
    log: Optional[bool]
    base: Optional[int]
    step: Optional[Union[int, float]]
    num_numeric_choices: Optional[int]


@dataclass
class OptimizeConifig:
    search_algorithm: str
    goal: List[OptimizerDirection]
    trial_number: int
    rand_seed: int
    sobol_scramble: bool
    grid_accept_small_trial_number: bool
    grid_sampling_method: str
    parameters: List[ParameterConfig]


@dataclass
class JobConfig:
    cancel_retry: int
    cancel_timeout: int
    expire_retry: int
    expire_timeout: int
    finished_retry: int
    finished_timeout: int
    job_retry: int
    job_timeout: int
    kill_retry: int
    kill_timeout: int
    result_retry: int
    runner_retry: int
    runner_timeout: int
    running_retry: int
    running_timeout: int
    init_fail_count: int
    name_length: int
    random_scheduling: bool


@dataclass
class LoggingItemConifig:
    master: str
    optimizer: str
    scheduler: str


@dataclass
class LoggerConfig:
    file: LoggingItemConifig
    log_level: LoggingItemConifig
    stream_level: LoggingItemConifig


@dataclass
class ConditionConfig:
    loop: int
    minimum: Union[float, int]
    maximum: Union[float, int]
    passed: Optional[bool]
    best: Optional[Union[float, int]]


@dataclass
class Config:
    generic: GenericConfig
    resource: ResourceConifig
    ABCI: AbciConifig
    optimize: OptimizeConifig
    job_setting: JobConfig
    logger: Optional[LoggerConfig]
    clean: Optional[bool]
    resume: Optional[Union[None, int]]
    config_path: Optional[Union[None, Path, str]]


def set_structs_false(conf: Container) -> None:
    OmegaConf.set_struct(conf, False)
    if hasattr(conf, "__iter__"):
        for item in conf:
            if isinstance(conf.__dict__["_content"], dict):
                set_structs_false(conf.__dict__["_content"][item])


def load_config(config_path: Path | str) -> DictConfig:
    """
    Load any configuration files, return the DictConfig object.
    Args:
        config_path (str): A path to a configuration file.
    Returns:
        config: DictConfig object
    """
    path = Path(config_path).resolve()

    if not path.exists():
        raise ValueError("Config is not found.")

    base = OmegaConf.structured(Config)
    default = OmegaConf.create(read_text("aiaccel.experimental.mpi", "default_config.yaml"))
    customize = OmegaConf.load(path)
    customize.config_path = str(path)
    if not isinstance(customize.optimize.goal, ListConfig):
        customize.optimize.goal = ListConfig([customize.optimize.goal])

    config: Union[ListConfig, DictConfig] = OmegaConf.merge(base, default)
    set_structs_false(config)
    config = OmegaConf.merge(config, customize)
    if not isinstance(config, DictConfig):
        raise RuntimeError("The configuration is not a DictConfig object.")
    return config


def is_multi_objective(config: DictConfig) -> bool:
    """Is the multi-objective option set in the configuration.

    Args:
        config (Config): A configuration object.

    Returns:
        bool: Is the multi--objective option set in the configuration or not.
    """
    return isinstance(config.optimize.goal, ListConfig) and len(config.optimize.goal) > 1
