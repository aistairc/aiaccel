from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from importlib.resources import read_text
from pathlib import Path
from typing import Any, List, Optional, Union

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
from omegaconf.base import Container


class ResourceType(Enum):
    abci: str = 'abci'
    local: str = 'local'
    python_local: str = 'python_local'

    @classmethod
    def _missing_(cls, value: Any) -> Any | None:
        value = value.lower()
        for member in cls:
            if member.value == value:
                return member
        return None


class OptimizerDirection(Enum):
    minimize: str = 'minimize'
    maximize: str = 'maximize'

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


@dataclass
class ResourceConifig:
    type: ResourceType
    num_node: int


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
    q: Optional[Union[float, int]]
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
class VerificationConfig:
    is_verified: bool
    condition: List[ConditionConfig]


@dataclass
class Config:
    generic: GenericConfig
    resource: ResourceConifig
    ABCI: AbciConifig
    optimize: OptimizeConifig
    job_setting: JobConfig
    logger: Optional[LoggerConfig]
    verification: Optional[VerificationConfig]
    clean: Optional[bool]
    resume: Optional[Union[None, int]]
    config_path: Optional[Union[None, Path, str]]


def set_structs_false(conf: Container) -> None:
    OmegaConf.set_struct(conf, False)
    if hasattr(conf, "__iter__"):
        for item in conf:
            if isinstance(conf.__dict__["_content"], dict):
                set_structs_false(conf.__dict__["_content"][item])


def load_config(config_path: Path | str) -> Union[ListConfig, DictConfig]:
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
    default = OmegaConf.create(read_text('aiaccel', 'default_config.yaml'))
    customize = OmegaConf.load(path)
    customize.config_path = str(path)
    if not isinstance(customize.optimize.goal, ListConfig):
        customize.optimize.goal = ListConfig([customize.optimize.goal])

    config: Union[ListConfig, DictConfig] = OmegaConf.merge(base, default)
    set_structs_false(config)
    config = OmegaConf.merge(config, customize)

    return config

<<<<<<< HEAD
class ConfigEntry:
    """ A class for defining values in a configuration file \
        or for holding read values.

    Args:
        config_path (str): A path of configuration file.
        type (list): A data type.
        default (Any): A default value.
        warning (bool): A flag of print a warning or not.
        group (str): A name of the group to which the parameter belongs.
        keys (tuple): A key to access the value For example,
            a parameter under 'generic' would be written as ('generic')

    Example:
        ::

            workspace = ConfigEntry(
                config=config,
                type=[str],
                default=_DEFAULT_WORKSPACE,
                warning=warn,
                group="generic",
                keys=("workspace")
            )
            workspace.get()

    """

    def __init__(
        self,
        config: ConfileWrapper,
        type: list,
        default: Any,
        warning: bool,
        group: str,
        keys: list | tuple | str
    ) -> None:
        self.config = config
        self.group = group
        self.type = type
        self.default = default
        self.warning = warning  # If use default, display warning
        self.group = group
        self.keys = keys
        self._value = None
        self.read_values_from_config_file = False

        # laod
        self.load_config_values()

    def get(self) -> Any:
        """
        Returns:
            self._value
        """
        return copy.deepcopy(self._value)

    def set(self, value) -> None:
        """
        Args
            value (any)
        """
        if type(value) not in self.type:
            logger.error(f"may be invalid value '{value}'.")
            raise TypeError
        self._value = value

    def show_warning(self) -> None:
        """ If the default value is used, a warning is displayed.
        """
        if self.warning:
            item: list[Any] = []
            item.append(self.group)
            if (
                type(self.keys) is list or
                type(self.keys) is tuple
            ):
                for key in self.keys:
                    item.append(key)
            else:
                item.append(self.keys)

            logger.warning(
                f"{'.'.join(item)} is not found in the configuration file, "
                f"the default value will be applied.(default: {self.default})"
            )

    def empty_if_error(self):
        """ If the value is not set, it will force an error to occur.
        """
        if (
            self._value is None or
            self._value == "" or
            self._value == [] or
            self._value == ()
        ):
            logger.error(f"Configuration error. '{self.keys}' is not found.")
            sys.exit()

    def load_config_values(self):
        """ Reads values from the configuration file.
        """
        if (
            type(self.keys) is list or
            type(self.keys) is tuple
        ):
            value = self.config.get(self.group, *self.keys)
        else:
            value = self.config.get(self.group, self.keys)

        if value is None:
            self.show_warning()
            self.set(self.default)
            self.read_values_from_config_file = False
        else:
            self.set(value)
            self.read_values_from_config_file = True

    @ property
    def Value(self):
        return self._value


# === Config file definition area ===

_DEFAULT_INIT_FAIL_COUNT = 100
_DEFAULT_NAME_LENGTH = 6
_DEFAULT_RAND_SEED = None
_DEFAULT_WORKSPACE = "./work"
_DEFAULT_MASTER_LOGFILE = "master.log"
_DEFAULT_MASTER_FILE_LOG_LEVEL = "DEBUG"
_DEFAULT_MASTER_STREAM_LOG_LEVEL = "DEBUG"
_DEFAULT_OPTIMIZER_LOGFILE = "optimizer.log"
_DEFAULT_OPTIMIZER_FILE_LOG_LEVEL = "DEBUG"
_DEFAULT_OPTIMIZER_STREAM_LOG_LEBEL = "DEBUG"
_DEFAULT_SCHDULER_LOGFILE = "scheduler.log"
_DEFAULT_SCHDULER_FILE_LOG_LEVEL = "DEBUG"
_DEFAULT_SCHDULER_STREAM_LOG_LEBEL = "DEBUG"
_DEFAULT_JOB_COMMAND = ""
_DEFAULT_SLEEP_TIME = 0.01
_DEFAULT_SEARCH_ALGORITHM = search_algorithm_nelder_mead
_DEFAULT_CANCEL_RETRY = 3
_DEFAULT_CANCEL_TIMEOUT = 60
_DEFAULT_EXPIRE_RETRY = 3
_DEFAULT_EXPIRE_TIMEOUT = 60
_DEFAULT_FINISHED_RETRY = 3
_DEFAULT_FINISHED_TIMEOUT = 60
_DEFAULT_JOB_LOOP_DURATION = 0.5
_DEFAULT_JOB_RETRY = 2
_DEFAULT_JOB_TIMEOUT = 60
_DEFAULT_KILL_RETRY = 3
_DEFAULT_KILL_TIMEOUT = 60
_DEFAULT_RESULT_RETRY = 1
_DEFAULT_BATCH_JOB_TIMEOUT = 600
_DEFAULT_RUNNER_RETRY = 3
_DEFAULT_RUNNER_TIMEOUT = 60
_DEFAULT_RUNNING_RETRY = 3
_DEFAULT_RUNNING_TIMEOUT = 60
_DEFAULT_RESOURCE_TYPE = "local"
_DEFAULT_NUM_NODE = 1
_DEFAULT_JOB_SCRIPT_PREAMBLE = ""
_DEFAULT_ABCI_GROUP = ""
_DEFAULT_JOB_EXECUTION_OPTIONS = ""
_DEFAULT_GOAL = "minimize"
_DEFAULT_MAX_TRIAL_NUMBER = 0
_DEFAULT_HYPERPARAMETERS: list = []
_DEFAULT_IS_VERIFIED = False
_DEFAULT_VERIFI_CONDITION: list = []
_DEFAULT_RANDOM_SCHESULING = True
_DEFAULT_SOBOL_SCRAMBLE = True
_DEFAULT_PYTHON_FILE = ""
_DEFAULT_FUNCTION = ""
_DEFAULT_IS_IGNORE_WARNING = True
=======
>>>>>>> 72450882f5821b5eec9e90cf5401df1d623b8443

def is_multi_objective(config: DictConfig) -> bool:
    """Is the multi-objective option set in the configuration.

    Args:
        config (Config): A configuration object.

<<<<<<< HEAD
    def __init__(
        self,
        config_path: str | Path,
        warn: bool = False,
        format_check: bool = False
    ):
        self.config_path = Path(config_path).resolve()
        if not self.config_path.exists():
            logger.error(f"config file: {config_path} doesn't exist.")

        self.config = load_config(config_path)
        self.define_items(self.config, warn)
        if format_check:
            self.workspace.empty_if_error()

            self.job_command.empty_if_error()

            if self.resource_type.get().lower() == "abci":
                self.abci_group.empty_if_error()
                self.job_script_preamble.empty_if_error()
                if Path(self.job_script_preamble.get()).exists() is False:
                    logger.error(f"{self.job_script_preamble.get()} is not found.")
                    sys.exit()
            # self.hps_format_check()

    def define_items(self, config: ConfileWrapper, warn: bool) -> None:
        """ Define the configuration of the configuration file

        Args:
            config (ConfileWrapper):
            warn (bool): A flag of print a warning or not. Defaults to
                False.
        """
        self.workspace = ConfigEntry(
            config=config,
            type=[str],
            default=_DEFAULT_WORKSPACE,
            warning=warn,
            group="generic",
            keys=("workspace")
        )
        self.job_command = ConfigEntry(
            config=config,
            type=[str],
            default=_DEFAULT_JOB_COMMAND,
            warning=warn,
            group="generic",
            keys=("job_command")
        )
        self.sleep_time = ConfigEntry(
            config=config,
            type=[float, int],
            default=_DEFAULT_SLEEP_TIME,
            warning=False,
            group="generic",
            keys=("sleep_time")
        )
        self.python_file = ConfigEntry(
            config=config,
            type=[str],
            default=_DEFAULT_PYTHON_FILE,
            warning=False,
            group="generic",
            keys=("python_file")
        )
        self.function = ConfigEntry(
            config=config,
            type=[str],
            default=_DEFAULT_FUNCTION,
            warning=False,
            group="generic",
            keys=("function")
        )

        # === scheduler defalt config===
        self.cancel_retry = ConfigEntry(
            config=config,
            type=[int],
            default=_DEFAULT_CANCEL_RETRY,
            warning=False,
            group="job_setting",
            keys=("cancel_retry")
        )
        self.cancel_timeout = ConfigEntry(
            config=config,
            type=[int],
            default=_DEFAULT_CANCEL_TIMEOUT,
            warning=False,
            group="job_setting",
            keys=("cancel_timeout")
        )
        self.expire_retry = ConfigEntry(
            config=config,
            type=[int],
            default=_DEFAULT_EXPIRE_RETRY,
            warning=False,
            group="job_setting",
            keys=("expire_retry")
        )
        self.expire_timeout = ConfigEntry(
            config=config,
            type=[int],
            default=_DEFAULT_EXPIRE_TIMEOUT,
            warning=False,
            group="job_setting",
            keys=("expire_timeout")
        )
        self.finished_retry = ConfigEntry(
            config=config,
            type=[int],
            default=_DEFAULT_FINISHED_RETRY,
            warning=False,
            group="job_setting",
            keys=("finished_retry")
        )
        self.finished_timeout = ConfigEntry(
            config=config,
            type=[int],
            default=_DEFAULT_FINISHED_TIMEOUT,
            warning=False,
            group="job_setting",
            keys=("finished_timeout")
        )
        self.job_loop_duration = ConfigEntry(
            config=config,
            type=[float],
            default=_DEFAULT_JOB_LOOP_DURATION,
            warning=False,
            group="job_setting",
            keys=("job_loop_duration")
        )
        self.job_retry = ConfigEntry(
            config=config,
            type=[int],
            default=_DEFAULT_JOB_RETRY,
            warning=False,
            group="job_setting",
            keys=("job_retry")
        )
        self.job_timeout = ConfigEntry(
            config=config,
            type=[int],
            default=_DEFAULT_JOB_TIMEOUT,
            warning=False,
            group="job_setting",
            keys=("job_timeout")
        )
        self.kill_retry = ConfigEntry(
            config=config,
            type=[int],
            default=_DEFAULT_KILL_RETRY,
            warning=False,
            group="job_setting",
            keys=("kill_retry")
        )
        self.kill_timeout = ConfigEntry(
            config=config,
            type=[int],
            default=_DEFAULT_KILL_TIMEOUT,
            warning=False,
            group="job_setting",
            keys=("kill_timeout")
        )
        self.result_retry = ConfigEntry(
            config=config,
            type=[int],
            default=_DEFAULT_RESULT_RETRY,
            warning=False,
            group="job_setting",
            keys=("result_retry")
        )
        self.batch_job_timeout = ConfigEntry(
            config=config,
            type=[int],
            default=_DEFAULT_BATCH_JOB_TIMEOUT,
            warning=False,
            group="generic",
            keys=("batch_job_timeout")
        )
        self.runner_retry = ConfigEntry(
            config=config,
            type=[int],
            default=_DEFAULT_RUNNER_RETRY,
            warning=False,
            group="job_setting",
            keys=("runner_retry")
        )
        self.runner_timeout = ConfigEntry(
            config=config,
            type=[int],
            default=_DEFAULT_RUNNER_TIMEOUT,
            warning=False,
            group="job_setting",
            keys=("runner_timeout")
        )
        self.running_retry = ConfigEntry(
            config=config,
            type=[int],
            default=_DEFAULT_RUNNING_RETRY,
            warning=False,
            group="job_setting",
            keys=("running_retry")
        )
        self.running_timeout = ConfigEntry(
            config=config,
            type=[int],
            default=_DEFAULT_RUNNING_TIMEOUT,
            warning=False,
            group="job_setting",
            keys=("running_timeout")
        )
        # === generic defalt config===
        self.init_fail_count = ConfigEntry(
            config=config,
            type=[int],
            default=_DEFAULT_INIT_FAIL_COUNT,
            warning=False,
            group="job_setting",
            keys=("init_fail_count"),
        )
        self.name_length = ConfigEntry(
            config=config,
            type=[int],
            default=_DEFAULT_NAME_LENGTH,
            warning=False,
            group="job_setting",
            keys=("name_length")
        )
        self.is_ignore_warning = ConfigEntry(
            config=config,
            type=[bool],
            default=_DEFAULT_IS_IGNORE_WARNING,
            warning=False,
            group="generic",
            keys=("is_ignore_warning")
        )
        # === resource defalt config ===
        self.resource_type = ConfigEntry(
            config=config,
            type=[str],
            default=_DEFAULT_RESOURCE_TYPE,
            warning=warn,
            group="resource",
            keys=("type")
        )
        self.num_node = ConfigEntry(
            config=config,
            type=[int],
            default=_DEFAULT_NUM_NODE,
            warning=warn,
            group="resource",
            keys=("num_node")
        )

        # === ABCI defalt config ===
        self.job_script_preamble = ConfigEntry(
            config=config,
            type=[str],
            default=_DEFAULT_JOB_SCRIPT_PREAMBLE,
            warning=warn,
            group="ABCI",
            keys=("job_script_preamble")
        )
        self.abci_group = ConfigEntry(
            config=config,
            type=[str],
            default=_DEFAULT_ABCI_GROUP,
            warning=warn,
            group="ABCI",
            keys=("group")
        )
        self.job_execution_options = ConfigEntry(
            config=config,
            type=[str],
            default=_DEFAULT_JOB_EXECUTION_OPTIONS,
            warning=warn,
            group="ABCI",
            keys=("job_execution_options")
        )

        # === hyperparameter defalt config ===
        self.goal = ConfigEntry(
            config=config,
            type=[str],
            default=_DEFAULT_GOAL,
            warning=warn,
            group="optimize",
            keys=("goal")
        )
        self.trial_number = ConfigEntry(
            config=config,
            type=[int],
            default=_DEFAULT_MAX_TRIAL_NUMBER,
            warning=warn,
            group="optimize",
            keys=("trial_number")
        )
        self.randseed = ConfigEntry(
            config=config,
            type=[int, NoneType],
            default=_DEFAULT_RAND_SEED,
            warning=warn,
            group="optimize",
            keys=("rand_seed")
        )
        self.hyperparameters = ConfigEntry(
            config=config,
            type=[list],
            default=_DEFAULT_HYPERPARAMETERS,
            warning=warn,
            group="optimize",
            keys=("parameters")
        )
        self.search_algorithm = ConfigEntry(
            config=config,
            type=[str],
            default=_DEFAULT_SEARCH_ALGORITHM,
            warning=warn,
            group="optimize",
            keys=("search_algorithm")
        )

        # === logger defalt config===
        self.master_logfile = ConfigEntry(
            config=config,
            type=[str],
            default=_DEFAULT_MASTER_LOGFILE,
            warning=False,
            group="logger",
            keys=("file", "master")
        )
        self.master_file_log_level = ConfigEntry(
            config=config,
            type=[str],
            default=_DEFAULT_MASTER_FILE_LOG_LEVEL,
            warning=False,
            group="logger",
            keys=("log_level", "master")
        )
        self.master_stream_log_level = ConfigEntry(
            config=config,
            type=[str],
            default=_DEFAULT_MASTER_STREAM_LOG_LEVEL,
            warning=False,
            group="logger",
            keys=("stream_level", "master")
        )
        self.optimizer_logfile = ConfigEntry(
            config=config,
            type=[str],
            default=_DEFAULT_OPTIMIZER_LOGFILE,
            warning=False,
            group="logger",
            keys=("file", "optimizer")
        )
        self.optimizer_file_log_level = ConfigEntry(
            config=config,
            type=[str],
            default=_DEFAULT_OPTIMIZER_FILE_LOG_LEVEL,
            warning=False,
            group="logger",
            keys=("log_level", "optimizer")
        )
        self.optimizer_stream_log_level = ConfigEntry(
            config=config,
            type=[str],
            default=_DEFAULT_OPTIMIZER_STREAM_LOG_LEBEL,
            warning=False,
            group="logger",
            keys=("stream_level", "optimizer")
        )
        self.scheduler_logfile = ConfigEntry(
            config=config,
            type=[str],
            default=_DEFAULT_SCHDULER_LOGFILE,
            warning=False,
            group="logger",
            keys=("file", "scheduler")
        )
        self.scheduler_file_log_level = ConfigEntry(
            config=config,
            type=[str],
            default=_DEFAULT_SCHDULER_FILE_LOG_LEVEL,
            warning=False,
            group="logger",
            keys=("log_level", "scheduler")
        )
        self.scheduler_stream_log_level = ConfigEntry(
            config=config,
            type=[str],
            default=_DEFAULT_SCHDULER_STREAM_LOG_LEBEL,
            warning=False,
            group="logger",
            keys=("stream_level", "scheduler")
        )

        self.sobol_scramble = ConfigEntry(
            config=config,
            type=[bool],
            default=_DEFAULT_SOBOL_SCRAMBLE,
            warning=False,
            group="optimize",
            keys=("sobol_scramble")
        )

        # === verification ===
        self.is_verified = ConfigEntry(
            config=config,
            type=[bool],
            default=_DEFAULT_IS_VERIFIED,
            warning=False,
            group="verification",
            keys=("is_verified")
        )
        self.condition = ConfigEntry(
            config=config,
            type=[list],
            default=_DEFAULT_VERIFI_CONDITION,
            warning=False,
            group="verification",
            keys=("condition")
        )
=======
    Returns:
        bool: Is the multi--objective option set in the configuration or not.
    """
    return isinstance(config.optimize.goal, ListConfig) and len(config.optimize.goal) > 1
>>>>>>> 72450882f5821b5eec9e90cf5401df1d623b8443
