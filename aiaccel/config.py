from dataclasses import dataclass
from importlib.resources import read_text
from pathlib import Path
from typing import Union, List, Optional, Any

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig


@dataclass
class GenericConfig:
    workspace: str
    job_command: str
    python_file: str
    function: str
    batch_job_timeout: int
    sleep_time: Union[float, int]

@dataclass
class ResourceConifig:
    type: str
    num_node: int

@dataclass
class AbciConifig:
    group: str
    job_script_preamble: str
    job_execution_options: str
    runner_search_pattern: str


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
    q: Optional[Union[float, int]]
    log: Optional[bool]
    base: Optional[int]
    step: Optional[Union[int, float]]


@dataclass
class OptimizeConifig:
    search_algorithm: str
    goal: str
    trial_number: int
    rand_seed: int
    sobol_scramble: bool
    parameters: List
    parameters: List[ParameterConfig]

@dataclass
class JobConifig:
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
@dataclass
class VerificationConfig:
    is_verified: bool
    condition: List[ConditionConfig]

@dataclass
class Config:
    generic: GenericConfig
    resource: ResourceConifig
    ABCI: AbciConifig
    optimize: OptimizeConifig
    job_setting: JobConifig
    logger: LoggerConfig
    verification: VerificationConfig
    clean: Optional[bool]
    resume: Optional[Union[None, int, bool]]
    config_path: Optional[Union[None, Path, str]]

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

    base = OmegaConf.structured(Config)
    default = OmegaConf.create(read_text('aiaccel', 'default_config.yaml'))
    customize = OmegaConf.load(path)
    customize.config_path = str(path)

    config = OmegaConf.merge(base, default, customize)

    return config
