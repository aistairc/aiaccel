import copy
import os
import sys
from abc import ABCMeta, abstractmethod
from logging import StreamHandler, getLogger
from pathlib import Path
from typing import Any, Union

import confile

from aiaccel.common import search_algorithm_nelder_mead

NoneType = type(None)

logger = getLogger(__name__)
logger.setLevel(os.getenv('LOG_LEVEL', 'INFO'))
logger.addHandler(StreamHandler())


class BaseConfig(metaclass=ABCMeta):
    """
    Fork by confile: https://github.com/777nancy/confile

    This is an interface for all config classes.
    """

    @abstractmethod
    def get_property(self, key, *keys):
        pass

    @abstractmethod
    def to_dict(self):
        pass


class JsonOrYamlObjectConfig(BaseConfig):
    """
    Fork by confile: https://github.com/777nancy/confile
    """

    def __init__(self, config: dict, file_type: str) -> None:
        """A wrapper for confile to support json, yaml object.

        Args:
            config (dict): A json or yaml object
            file_type (str): 'json_object' or 'yaml_object'
        """
        if file_type in ['json_object', 'yaml_object']:
            self._config_dict = config
        else:
            raise TypeError(f'Unknown file type {file_type}')

    def get_property(self, key: str, *keys: str) ->\
            Union[str, list, dict, None]:
        """
        Get a property for specified keys.

        Args:
            key (str): A key to get a property.
            *keys (str): Keys to get a property.

        Returns:
            Union[str, list, dict, None]: A property for the keys.
        """
        if type(self._config_dict) is list:
            return None

        sub_config_dict = self._config_dict.get(key)

        if keys and sub_config_dict is not None:
            for k in keys:
                if type(sub_config_dict) is not dict or\
                        sub_config_dict is None:
                    return None
                value = sub_config_dict.get(k)
                sub_config_dict = value
            return sub_config_dict
        else:
            return sub_config_dict

    def to_dict(self) -> dict:
        """
        Convert the configuration to a dictionary object.

        Returns:
            dict: The dictionary object of the configuration.
        """
        return self._config_dict


class ConfileWrapper(object):
    """
    A wrapper class for confile library.

    Thins wrapper class supports to load a configuration file in JSON object,
    JSON file and YAML format. It provides a simple method 'get' to get a
    property for the specified keys.
    """

    def __init__(self, config: Any, config_type: str) -> None:
        """

        Args:
            config (Any): A file path to configuration file.
            config_type (str): A file path to default configuration file.
        """
        config_types = [
            'json_file',
            'yaml_file',
            'json_object',
            'yaml_object'
        ]
        if config_type not in config_types:
            raise TypeError(f'Unknown config type: {config_type}')

        if config_type in ['json_file', 'yaml_file']:
            self.config = confile.read_config(str(config))
        elif config_type in ['json_object', 'yaml_object']:
            self.config = JsonOrYamlObjectConfig(config, config_type)

    def get(self, key: str, *keys: str) -> Union[str, list, dict, None]:
        """Get a property with specified keys.

        Args:
            key (str): A key for the property
            *keys (list): Nested eys for the property

        Returns:
            Union[str, list, dict, None]: A property for the specified keys.
        """
        p = self.config.get_property(key, *keys)
        return p


def load_config(config_path: str) -> ConfileWrapper:
    """
    Load any configuration files, return the ConfileWrapper object.
    Args:
        config_path (str): A path to a configuration file.

    Returns:
        ConfileWrapper: A wrapper object of the configuration.
    """
    path = Path(config_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f'config file cannot be found: {config_path}')

    file_type = path.suffix[1:].lower()

    if file_type == 'json':
        return ConfileWrapper(config_path, 'json_file')
    elif file_type in ['yml', 'yaml']:
        return ConfileWrapper(config_path, 'yaml_file')
    else:
        raise TypeError(f'Unknown file type {file_type}')


class ConfigEntry:
    """ A class for defining values in a configuration file
        or for holding read values.

    Exmple:
        ```
        workspace = ConfigEntry(
            config=config,
            type=[str],
            default=_DEFAULT_WORKSPACE,
            warning=warn,
            group="generic",
            keys=("workspace")
        )
        workspace.get()
        ```

    """

    def __init__(
        self,
        config: ConfileWrapper,
        type: list,
        default: Any,
        warning: bool,
        group: str,
        keys: tuple
    ):
        """
        Args:
            config_path (str): A path of configuration file.
            type (list): A data type.
            default (any): A default value.
            warning (bool): A flag of print a warning or not.
            group (str): A name of the group to which the parameter belongs.
            keys (tuple): A key to access the value For example,
                a parameter under 'generic' would be written as ('generic')

        Returns:
            None

        """
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

    def set(self, value):
        """
        Args
            value (any)
        """
        if type(value) not in self.type:
            logger.error(f"may be invalid value '{value}'.")
            raise TypeError
        self._value = value

    def show_warning(self):
        """ If the default value is used, a warning is displayed.
        """
        if self.warning:
            item = []
            item.append(self.group)
            if (
                type(self.keys) is list or
                type(self.keys) is tuple
            ):
                for key in self.keys:
                    item.append(key)
            else:
                item.append(self.keys)

            item = ".".join(item)
            logger.warning(
                f"{item} is not found in the configuration file, "
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

    @property
    def Value(self):
        return self._value


# === Config file definition area ===

_DEFAULT_INIT_FAIL_COUNT = 100
_DEFAULT_NAME_LENGTH = 6
_DEFAULT_RAND_SEED = None
_DEFAULT_SILENT_MODE = True
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
_DEFAULT_RUNNER_SEARCH_PATTERN = "run_*.sh"
_DEFAULT_GOAL = "minimize"
_DEFAULT_MAX_TRIAL_NUMBER = 0
_DEFAULT_HYPERPARAMETERS = []
_DEFAULT_IS_VERIFIED = False
_DEFAULT_VERIFI_CONDITION = []
_DEFAULT_RANDOM_SCHESULING = True
_RESOURCE_TYPES = ['abci', 'local']
_GOALS = ['minimize', 'maximize']


class Config:
    """ A Class for defining the configuration of a configuration file.
    """

    def __init__(
        self,
        config_path: str,
        warn=False,
        format_check=False
    ):
        """
        Args:
            config_path (str): A path of configuration file.
            warn (bool): A flag of print a warning or not.
            format_check (bool): A flag of do tha check format or not.
        """
        self.config_path = Path(config_path).resolve()
        if not self.config_path.exists():
            logger.erro(f"config file: {config_path} doesn't exist.")

        self.config = load_config(self.config_path)
        self.define_items(self.config, warn)
        if format_check:
            self.workspace.empty_if_error()

            self.job_command.empty_if_error()

            if self.goal.get().lower() not in _GOALS:
                logger.error(f'Invalid goal: {self.goal.get()}')

            if self.resource_type.get().lower() not in _RESOURCE_TYPES:
                logger.error(f'Invalid resource type: {self.resource_type.get()}.')
                sys.exit()

            if self.resource_type.get().lower() == "abci":
                self.abci_group.empty_if_error()
                self.job_script_preamble.empty_if_error()
                if Path(self.job_script_preamble.get()).exists() is False:
                    logger.error(f"{self.job_script_preamble.get()} is not found.")
                    sys.exit()
            # self.hps_format_check()

    def define_items(self, config: ConfileWrapper, warn: bool):
        """ Define the configuration of the configuration file
        """
        self.silent_mode = ConfigEntry(
            config=config,
            type=[bool],
            default=_DEFAULT_SILENT_MODE,
            warning=False,
            group="ui",
            keys=("silent_mode")
        )
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
        # Randomize the scheduling (default=False).
        # This is probably not needed, but we'll keep it just in case.
        # random_scheduling = True for random scheduling.
        self.random_scheduling = ConfigEntry(
            config=config,
            type=[bool],
            default=_DEFAULT_RANDOM_SCHESULING,
            warning=False,
            group="job_setting",
            keys=("random_scheduling")
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
        self.runner_search_pattern = ConfigEntry(
            config=config,
            type=[str],
            default=_DEFAULT_RUNNER_SEARCH_PATTERN,
            warning=False,
            group="ABCI",
            keys=("runner_search_pattern")
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
