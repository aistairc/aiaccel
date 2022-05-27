from abc import ABCMeta
from abc import abstractmethod
from pathlib import Path
from typing import Any, Union
import confile
import copy
from aiaccel.common import search_algorithm_grid
from aiaccel.common import search_algorithm_nelder_mead
from aiaccel.common import search_algorithm_random
from aiaccel.common import search_algorithm_sobol
from aiaccel.common import search_algorithm_tpe
from aiaccel.util.terminal import Terminal
from aiaccel.util.wd import get_num_node_all  # wd/
import sys
NoneType = type(None)


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
            raise TypeError('Unknown file type {}'.format(file_type))

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
            raise TypeError('Unknown config type: {}'.format(config_type))

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
        raise FileNotFoundError(
            'config file cannot be found: {}'
            .format(config_path)
        )

    file_type = path.suffix[1:].lower()

    if file_type == 'json':
        return ConfileWrapper(config_path, 'json_file')
    elif file_type in ['yml', 'yaml']:
        return ConfileWrapper(config_path, 'yaml_file')
    else:
        raise TypeError('Unknown file type {}'.format(file_type))


class ConfigEntry:
    """ A class for defining values in a configuration file
        or for holding read values.

    Exmple:
        ```
        workspace = ConfigEntry(
            config_file_path=config_file_path,
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
        config_file_path: str,
        type: list,
        default: Any,
        warning: bool,
        group: str,
        keys: tuple
    ):
        """
        Args:
            config_file_path (str): A path of configuration file.
            type (list): A data type.
            default (any): A default value.
            warning (bool): A flag of print a warning or not.
            group (str): A name of the group to which the parameter belongs.
            keys (tuple): A key to access the value For example,
                a parameter under 'generic' would be written as ('generic')

        Returns:
            None

        """
        self.config_file_path = config_file_path
        self.group = group
        self.type = type
        self.default = default
        self.warning = warning  # If use default, display warning
        self.group = group
        self.keys = keys
        self._value = None
        self.config = None
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
            Terminal().print_error(
                "may be invalid value '{}'.".format(value)
            )
            raise TypeError
        self._value = value

    def show_warning(self):
        """　If the default value is used, a warning is displayed.
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
            Terminal().print_warning(
                "{} is not found in the configuration file, "
                "the default value will be applied.(default: {})".format(
                    item, self.default
                )
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
            Terminal().print_error(
                "Configuration error. '{}' is not found."
                .format(self.keys)
            )
            sys.exit()

    def load_config_values(self):
        """ Reads values from the configuration file.
        """
        if self.config is None:
            self.config = load_config(self.config_file_path)

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

        self.config = None

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
_DEFAULT_SLEEP_TIME_MASTER = 1
_DEFAULT_OPTIMIZER_COMMAND = "python -m aiaccel.bin.optimizer"
_DEFAULT_SCHDULER_COMMAND = "python -m aiaccel.bin.scheduler"
_DEFAULT_SEARCH_ALGORITHM = search_algorithm_nelder_mead
_DEFAULT_SLEEP_TIME_OPTIMIZER = 1
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
_DEFAULT_SLEEP_TIME_SCHEDULER = 1
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
        config_file_path: str,
        warn=False,
        format_check=False
    ):
        """
        Args:
            config_file_path (str): A path of configuration file.
            warn (bool): A flag of print a warning or not.
            format_check (bool): A flag of do tha check format or not.
        """
        self.config_file_path = Path(config_file_path).resolve()
        self.define_items(config_file_path, warn)
        if format_check:
            self.workspace.empty_if_error()

            self.job_command.empty_if_error()

            if self.goal.get().lower() not in _GOALS:
                Terminal().print_error(
                    'Invalid goal: {}'
                    .format(self.goal.get())
                )

            if self.resource_type.get().lower() not in _RESOURCE_TYPES:
                Terminal().print_error(
                    'Invalid resource type: {}.'
                    .format(self.resource_type.get())
                )
                sys.exit()

            if self.resource_type.get().lower() == "abci":
                self.abci_group.empty_if_error()
                self.job_script_preamble.empty_if_error()
                if Path(self.job_script_preamble.get()).exists() is False:
                    Terminal().print_error(
                        "{} is not found."
                        .format(self.job_script_preamble.get())
                    )
                    sys.exit()
            self.hps_format_check()

    def define_items(self, config_file_path, warn):
        """ Define the configuration of the configuration file
        """

        self.silent_mode = ConfigEntry(
            config_file_path=config_file_path,
            type=[bool],
            default=_DEFAULT_SILENT_MODE,
            warning=False,
            group="ui",
            keys=("silent_mode")
        )
        self.workspace = ConfigEntry(
            config_file_path=config_file_path,
            type=[str],
            default=_DEFAULT_WORKSPACE,
            warning=warn,
            group="generic",
            keys=("workspace")
        )
        self.job_command = ConfigEntry(
            config_file_path=config_file_path,
            type=[str],
            default=_DEFAULT_JOB_COMMAND,
            warning=warn,
            group="generic",
            keys=("job_command")
        )
        self.optimizer_command = ConfigEntry(
            config_file_path=config_file_path,
            type=[str],
            default=_DEFAULT_OPTIMIZER_COMMAND,
            warning=False,
            group="generic",
            keys=("optimizer_command")
        )
        self.scheduler_command = ConfigEntry(
            config_file_path=config_file_path,
            type=[str],
            default=_DEFAULT_SCHDULER_COMMAND,
            warning=False,
            group="generic",
            keys=("scheduler_command")
        )

        # === scheduler defalt config===
        self.cancel_retry = ConfigEntry(
            config_file_path=config_file_path,
            type=[int],
            default=_DEFAULT_CANCEL_RETRY,
            warning=False,
            group="job_setting",
            keys=("cancel_retry")
        )
        self.cancel_timeout = ConfigEntry(
            config_file_path=config_file_path,
            type=[int],
            default=_DEFAULT_CANCEL_TIMEOUT,
            warning=False,
            group="job_setting",
            keys=("cancel_timeout")
        )
        self.expire_retry = ConfigEntry(
            config_file_path=config_file_path,
            type=[int],
            default=_DEFAULT_EXPIRE_RETRY,
            warning=False,
            group="job_setting",
            keys=("expire_retry")
        )
        self.expire_timeout = ConfigEntry(
            config_file_path=config_file_path,
            type=[int],
            default=_DEFAULT_EXPIRE_TIMEOUT,
            warning=False,
            group="job_setting",
            keys=("expire_timeout")
        )
        self.finished_retry = ConfigEntry(
            config_file_path=config_file_path,
            type=[int],
            default=_DEFAULT_FINISHED_RETRY,
            warning=False,
            group="job_setting",
            keys=("finished_retry")
        )
        self.finished_timeout = ConfigEntry(
            config_file_path=config_file_path,
            type=[int],
            default=_DEFAULT_FINISHED_TIMEOUT,
            warning=False,
            group="job_setting",
            keys=("finished_timeout")
        )
        self.job_loop_duration = ConfigEntry(
            config_file_path=config_file_path,
            type=[float],
            default=_DEFAULT_JOB_LOOP_DURATION,
            warning=False,
            group="job_setting",
            keys=("job_loop_duration")
        )
        self.job_retry = ConfigEntry(
            config_file_path=config_file_path,
            type=[int],
            default=_DEFAULT_JOB_RETRY,
            warning=False,
            group="job_setting",
            keys=("job_retry")
        )
        self.job_timeout = ConfigEntry(
            config_file_path=config_file_path,
            type=[int],
            default=_DEFAULT_JOB_TIMEOUT,
            warning=False,
            group="job_setting",
            keys=("job_timeout")
        )
        self.kill_retry = ConfigEntry(
            config_file_path=config_file_path,
            type=[int],
            default=_DEFAULT_KILL_RETRY,
            warning=False,
            group="job_setting",
            keys=("kill_retry")
        )
        self.kill_timeout = ConfigEntry(
            config_file_path=config_file_path,
            type=[int],
            default=_DEFAULT_KILL_TIMEOUT,
            warning=False,
            group="job_setting",
            keys=("kill_timeout")
        )
        self.result_retry = ConfigEntry(
            config_file_path=config_file_path,
            type=[int],
            default=_DEFAULT_RESULT_RETRY,
            warning=False,
            group="job_setting",
            keys=("result_retry")
        )
        self.batch_job_timeout = ConfigEntry(
            config_file_path=config_file_path,
            type=[int],
            default=_DEFAULT_BATCH_JOB_TIMEOUT,
            warning=False,
            group="generic",
            keys=("batch_job_timeout")
        )
        self.runner_retry = ConfigEntry(
            config_file_path=config_file_path,
            type=[int],
            default=_DEFAULT_RUNNER_RETRY,
            warning=False,
            group="job_setting",
            keys=("runner_retry")
        )
        self.runner_timeout = ConfigEntry(
            config_file_path=config_file_path,
            type=[int],
            default=_DEFAULT_RUNNER_TIMEOUT,
            warning=False,
            group="job_setting",
            keys=("runner_timeout")
        )
        self.running_retry = ConfigEntry(
            config_file_path=config_file_path,
            type=[int],
            default=_DEFAULT_RUNNING_RETRY,
            warning=False,
            group="job_setting",
            keys=("running_retry")
        )
        self.running_timeout = ConfigEntry(
            config_file_path=config_file_path,
            type=[int],
            default=_DEFAULT_RUNNING_TIMEOUT,
            warning=False,
            group="job_setting",
            keys=("running_timeout")
        )
        # === generic defalt config===
        self.init_fail_count = ConfigEntry(
            config_file_path=config_file_path,
            type=[int],
            default=_DEFAULT_INIT_FAIL_COUNT,
            warning=False,
            group="job_setting",
            keys=("init_fail_count"),
        )
        self.name_length = ConfigEntry(
            config_file_path=config_file_path,
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
            config_file_path=config_file_path,
            type=[bool],
            default=_DEFAULT_RANDOM_SCHESULING,
            warning=False,
            group="job_setting",
            keys=("random_scheduling")
        )

        # === resource defalt config ===
        self.resource_type = ConfigEntry(
            config_file_path=config_file_path,
            type=[str],
            default=_DEFAULT_RESOURCE_TYPE,
            warning=warn,
            group="resource",
            keys=("type")
        )
        self.num_node = ConfigEntry(
            config_file_path=config_file_path,
            type=[int],
            default=_DEFAULT_NUM_NODE,
            warning=warn,
            group="resource",
            keys=("num_node")
        )
        # wd/
        wd_num_node_all = get_num_node_all()
        if wd_num_node_all >= 1:
            self.resource_type.set('local')
            self.num_node.set(wd_num_node_all)

        # === ABCI defalt config ===
        self.job_script_preamble = ConfigEntry(
            config_file_path=config_file_path,
            type=[str],
            default=_DEFAULT_JOB_SCRIPT_PREAMBLE,
            warning=warn,
            group="ABCI",
            keys=("job_script_preamble")
        )
        self.abci_group = ConfigEntry(
            config_file_path=config_file_path,
            type=[str],
            default=_DEFAULT_ABCI_GROUP,
            warning=warn,
            group="ABCI",
            keys=("group")
        )
        self.job_execution_options = ConfigEntry(
            config_file_path=config_file_path,
            type=[str],
            default=_DEFAULT_JOB_EXECUTION_OPTIONS,
            warning=warn,
            group="ABCI",
            keys=("job_execution_options")
        )
        self.runner_search_pattern = ConfigEntry(
            config_file_path=config_file_path,
            type=[str],
            default=_DEFAULT_RUNNER_SEARCH_PATTERN,
            warning=False,
            group="ABCI",
            keys=("runner_search_pattern")
        )

        # === hyperparameter defalt config ===
        self.goal = ConfigEntry(
            config_file_path=config_file_path,
            type=[str],
            default=_DEFAULT_GOAL,
            warning=warn,
            group="optimize",
            keys=("goal")
        )
        self.trial_number = ConfigEntry(
            config_file_path=config_file_path,
            type=[int],
            default=_DEFAULT_MAX_TRIAL_NUMBER,
            warning=warn,
            group="optimize",
            keys=("trial_number")
        )
        self.randseed = ConfigEntry(
            config_file_path=config_file_path,
            type=[int, NoneType],
            default=_DEFAULT_RAND_SEED,
            warning=warn,
            group="optimize",
            keys=("rand_seed")
        )
        self.hyperparameters = ConfigEntry(
            config_file_path=config_file_path,
            type=[list],
            default=_DEFAULT_HYPERPARAMETERS,
            warning=warn,
            group="optimize",
            keys=("parameters")
        )
        self.search_algorithm = ConfigEntry(
            config_file_path=config_file_path,
            type=[str],
            default=_DEFAULT_SEARCH_ALGORITHM,
            warning=warn,
            group="optimize",
            keys=("search_algorithm")
        )

        # === sleep time ===
        self.sleep_time_master = ConfigEntry(
            config_file_path=config_file_path,
            type=[float, int],
            default=_DEFAULT_SLEEP_TIME_MASTER,
            warning=False,
            group="sleep_time",
            keys=("master")
        )
        self.sleep_time_scheduler = ConfigEntry(
            config_file_path=config_file_path,
            type=[float, int],
            default=_DEFAULT_SLEEP_TIME_SCHEDULER,
            warning=False,
            group="sleep_time",
            keys=("scheduler")
        )
        self.sleep_time_optimizer = ConfigEntry(
            config_file_path=config_file_path,
            type=[float, int],
            default=_DEFAULT_SLEEP_TIME_OPTIMIZER,
            warning=False,
            group="sleep_time",
            keys=("optimizer")
        )

        # === logger defalt config===
        self.master_logfile = ConfigEntry(
            config_file_path=config_file_path,
            type=[str],
            default=_DEFAULT_MASTER_LOGFILE,
            warning=False,
            group="logger",
            keys=("file", "master")
        )
        self.master_file_log_level = ConfigEntry(
            config_file_path=config_file_path,
            type=[str],
            default=_DEFAULT_MASTER_FILE_LOG_LEVEL,
            warning=False,
            group="logger",
            keys=("log_level", "master")
        )
        self.master_stream_log_level = ConfigEntry(
            config_file_path=config_file_path,
            type=[str],
            default=_DEFAULT_MASTER_STREAM_LOG_LEVEL,
            warning=False,
            group="logger",
            keys=("stream_level", "master")
        )
        self.optimizer_logfile = ConfigEntry(
            config_file_path=config_file_path,
            type=[str],
            default=_DEFAULT_OPTIMIZER_LOGFILE,
            warning=False,
            group="logger",
            keys=("file", "optimizer")
        )
        self.optimizer_file_log_level = ConfigEntry(
            config_file_path=config_file_path,
            type=[str],
            default=_DEFAULT_OPTIMIZER_FILE_LOG_LEVEL,
            warning=False,
            group="logger",
            keys=("log_level", "optimizer")
        )
        self.optimizer_stream_log_level = ConfigEntry(
            config_file_path=config_file_path,
            type=[str],
            default=_DEFAULT_OPTIMIZER_STREAM_LOG_LEBEL,
            warning=False,
            group="logger",
            keys=("stream_level", "optimizer")
        )
        self.scheduler_logfile = ConfigEntry(
            config_file_path=config_file_path,
            type=[str],
            default=_DEFAULT_SCHDULER_LOGFILE,
            warning=False,
            group="logger",
            keys=("file", "scheduler")
        )
        self.scheduler_file_log_level = ConfigEntry(
            config_file_path=config_file_path,
            type=[str],
            default=_DEFAULT_SCHDULER_FILE_LOG_LEVEL,
            warning=False,
            group="logger",
            keys=("log_level", "scheduler")
        )
        self.scheduler_stream_log_level = ConfigEntry(
            config_file_path=config_file_path,
            type=[str],
            default=_DEFAULT_SCHDULER_STREAM_LOG_LEBEL,
            warning=False,
            group="logger",
            keys=("stream_level", "scheduler")
        )

        # === verification ===
        self.is_verified = ConfigEntry(
            config_file_path=config_file_path,
            type=[bool],
            default=_DEFAULT_IS_VERIFIED,
            warning=False,
            group="verification",
            keys=("is_verified")
        )
        self.condition = ConfigEntry(
            config_file_path=config_file_path,
            type=[list],
            default=_DEFAULT_VERIFI_CONDITION,
            warning=False,
            group="verification",
            keys=("condition")
        )

    def hps_format_check(self):
        """ Check the hyperparameter items.
        Note
            Available items
            * Random: uniform_float, uniform_int, categorical, ordinal
            * Grid: uniform_float, uniform_int, categorical, ordinal
            * Sobol: uniform_float, uniform_int
            * NM: uniform_float, uniform_int
            * TPE: niform_float, uniform_int, categorical, ordinal
        """
        algorithm = self.search_algorithm.get()
        hyperparameters = self.hyperparameters.get()

        if hyperparameters == []:
            Terminal().print_error(
                "'hyperparameters' are empty."
            )
            sys.exit()

        # === item check (individual)===
        if algorithm.lower() == search_algorithm_random:
            self._check_random_setting_format(algorithm, hyperparameters)

        elif algorithm.lower() == search_algorithm_grid:
            self._check_grid_setting_format(algorithm, hyperparameters)

        elif algorithm.lower() == search_algorithm_sobol:
            self._check_sobol_setting_format(algorithm, hyperparameters)

        elif algorithm.lower() == search_algorithm_nelder_mead:
            self._check_neldermead_setting_format(algorithm, hyperparameters)

        elif algorithm.lower() == search_algorithm_tpe:
            self._check_tpe_setting_format(algorithm, hyperparameters)

        else:
            Terminal().print_error(
                "algorithm: {} is not suportted.\n"
                "       You can set 'random', 'grid', 'sobol', "
                "'nelder-mead', and 'tpe'"
                .format(algorithm)
            )
            sys.exit()

    def _check_random_setting_format(
        self,
        algorithm: str,
        hyperparameters: list
    ) -> None:
        """ Check the format when random seach.
        Args
            algorithm (str): A name of seach algorithm.
            hyperparameters (list): Items of hyperparametes
        Note
            Available items
            uniform_float, uniform_int, categorical, ordinal
        """
        hp_types = [
            'uniform_float',
            'uniform_int',
            'categorical',
            'ordinal'
        ]
        fmt = FormatChecker(algorithm, hp_types, hyperparameters)

        # int, float
        necessary_items = ["name", "type", "lower", "upper"]
        optional_items = ["initial", "comment"]
        fmt.check_uniform_int(necessary_items, optional_items)
        fmt.check_uniform_float(necessary_items, optional_items)

        # categorical
        necessary_items = ["name", "type", "choices"]
        optional_items = ["initial", "comment"]
        fmt.check_categorical(necessary_items, optional_items)

        # ordinal
        necessary_items = ["name", "type", "lower", "upper", "sequence"]
        optional_items = ["initial", "comment"]
        fmt.check_ordinal(necessary_items, optional_items)

        # initial check
        fmt.check_initial_type([int, float, str])

    def _check_grid_setting_format(
        self,
        algorithm: str,
        hyperparameters: list
    ) -> None:
        """ Check the format when grid search.

        Args:
            algorithm (str): A name of seach algorithm.
            hyperparameters (list): Items of hyperparametes
        Note
            Available items
            uniform_float, uniform_int, categorical, ordinal
        """
        hp_types = [
            'uniform_float',
            'uniform_int'
        ]
        fmt = FormatChecker(algorithm, hp_types, hyperparameters)

        # int, float
        necessary_items = [
            "name", "type", "lower", "upper", "log", "base", "step"
        ]
        optional_items = ["comment"]
        fmt.check_uniform_int(necessary_items, optional_items)
        fmt.check_uniform_float(necessary_items, optional_items)

        # categorical
        necessary_items = ["name", "type", "choices"]
        optional_items = ["comment"]
        fmt.check_categorical(necessary_items, optional_items)

        # ordinal
        necessary_items = ["name", "type", "lower", "upper", "sequence"]
        optional_items = ["comment"]
        fmt.check_ordinal(necessary_items, optional_items)

        # initial check
        fmt.check_initial_type([int, float, str])

    def _check_sobol_setting_format(self, algorithm, hyperparameters):
        """ Check the format when sobol search.

        Args:
            algorithm (str): A name of seach algorithm.
            hyperparameters (list): Items of hyperparametes.
        Note
            Available items
            * Sobol: uniform_float, uniform_int
        """

        #
        # (Issues #14)
        # https://gitlab.com/onishi-lab/opt/-/issues/14
        # The calculated value of hyperparameters set to int type
        # in sobol search becomes a float.
        # -> Int type is not supported.
        #

        hp_types = [
            'uniform_float'
        ]
        fmt = FormatChecker(algorithm, hp_types, hyperparameters)

        # int, float
        necessary_items = ["name", "type", "lower", "upper"]
        optional_items = ["initial", "comment"]
        fmt.check_uniform_int(necessary_items, optional_items)
        fmt.check_uniform_float(necessary_items, optional_items)

        # initial check
        fmt.check_initial_type([int, float])

    def _check_neldermead_setting_format(
        self,
        algorithm: str,
        hyperparameters: list
    ):
        """ Check the format when nelder-mead search.

        Args:
            algorithm (str): A name of seach algorithm.
            hyperparameters (list): Items of hyperparametes
        Note
            Available items
            uniform_float, uniform_int
        """
        hp_types = [
            'uniform_float',
            'uniform_int'
        ]
        fmt = FormatChecker(algorithm, hp_types, hyperparameters)

        # int, float
        necessary_items = ["name", "type", "lower", "upper"]
        optional_items = ["initial", "comment"]
        fmt.check_uniform_int(necessary_items, optional_items)
        fmt.check_uniform_float(necessary_items, optional_items)

        # initial check
        fmt.check_initial_type([int, float, list])

    def _check_tpe_setting_format(
        self,
        algorithm: str,
        hyperparameters: list
    ) -> None:
        """ Check the format when TPE search.
        Args
            algorithm (str): A name of seach algorithm.
            hyperparameters (list): Items of hyperparametes
        Note
            Available items
            niform_float, uniform_int, categorical, ordinal
        """
        hp_types = [
            'uniform_float',
            'uniform_int',
            'categorical',
            'ordinal'
        ]
        fmt = FormatChecker(algorithm, hp_types, hyperparameters)

        # int, float
        necessary_items = ["name", "type", "lower", "upper"]
        optional_items = ["initial", "comment"]
        fmt.check_uniform_int(necessary_items, optional_items)
        fmt.check_uniform_float(necessary_items, optional_items)

        # categorical
        necessary_items = ["name", "type", "choices"]
        optional_items = ["initial", "comment"]
        fmt.check_categorical(necessary_items, optional_items)

        # ordinal
        necessary_items = ["name", "type", "lower", "upper", "sequence"]
        optional_items = ["initial", "comment"]
        fmt.check_ordinal(necessary_items, optional_items)

        # initial check
        fmt.check_initial_type([int, float, str])


class FormatChecker:
    """ Configuration file format check

    Attributes:
        supprt_search_types (list):
            List of data types supported　by this search algorithm.
            * Items
                * uniform_float
                * uniform_int
                * categorical
                * ordinal
        hyperparameters (list): Items of hyperparametes.
        algorithm (str): search algorithm search algorithm.
    """

    def __init__(
        self,
        algorithm: str,
        supprt_search_types: list,
        hyperparameters: list
    ):
        self.supprt_search_types = supprt_search_types
        self.hyperparameters = hyperparameters
        self.algorithm = algorithm
        for hp in self.hyperparameters:
            if hp['type'] not in self.supprt_search_types:
                Terminal().print_error(
                    "'{}' is not support {}"
                    .format(self.algorithm, hp['type'])
                )
                sys.exit()

    def check_hyperparameters_item(
        self,
        hp: dict,
        necessary_items: list,
        optional_items: list
    ) -> bool:
        """ Check the hyperpaarmeters has necessary or optional items.

        Args:
            hp (dict),
            necessary_items (list): List of necessary items.
            optional_items (list):  List of optional items.
        """
        necessary = set(necessary_items)
        defines = set(hp.keys())
        not_found_items = list(necessary - defines)

        if len(not_found_items) > 0:
            for item in not_found_items:
                if item not in optional_items:
                    Terminal().print_error(
                        "Not found '{}' in 'hyperparameters'"
                        .format(item)
                    )
                    return False

        not_supported_items = list(defines - necessary)
        if len(not_supported_items) > 0:
            for item in not_supported_items:
                if item not in optional_items:
                    Terminal().print_error(
                        "'{}' is not supported in {} {}"
                        .format(item, self.algorithm, hp['type'])
                    )
                    return False
        return True

    def check_uniform_int(
        self,
        necessary_items: list,
        optional_items: list
    ) -> None:
        """ check_uniform_int

        Check the format of a hyperparameter
        when its data type is uniform int.

        Args:
            hp (dict): Items of hyperparameter.

            necessary_items (list): List of necessary items.
                example: ["name", "type", "lower", "upper"]

            optional_items (list):  List of optional items.
                example: ["initial", "comment"]
        """
        for hp in self.hyperparameters:
            if hp['type'] == 'uniform_int':
                if self.check_hyperparameters_item(
                    hp, necessary_items, optional_items
                ) is False:
                    sys.exit()

    def check_uniform_float(
        self,
        necessary_items: list,
        optional_items: list
    ) -> None:
        """ check_uniform_float

        Check the format of a hyperparameter when
        its data type is uniform float.

        Args:
            hp (dict): Items of hyperparameter.

            necessary_items (list): List of necessary items.
                example: ["name", "type", "lower", "upper"]

            optional_items (list):  List of optional items.
                example: ["initial", "comment"]
        """
        for hp in self.hyperparameters:
            if hp['type'] == 'uniform_float':
                if self.check_hyperparameters_item(
                    hp, necessary_items, optional_items
                ) is False:
                    sys.exit()

    def check_categorical(
        self,
        necessary_items: list,
        optional_items: list
    ) -> None:
        """ check_categorical

        Check the format of a hyperparameter when
        its data type is categorical.

        Args:
            hp (dict): Items of hyperparameter.

            necessary_items (list): List of necessary items.
                example: ["name", "type", "choices"]

            optional_items (list):  List of optional items.
                example: ["initial", "comment"]
        """
        for hp in self.hyperparameters:
            if hp['type'] == 'categorical':
                if self.check_hyperparameters_item(
                    hp, necessary_items, optional_items
                ) is False:
                    sys.exit()

    def check_ordinal(
        self,
        necessary_items: list,
        optional_items: list
    ) -> None:
        """ check_ordinal

        Check the format of a hyperparameter when
        its data type is ordinal.

        Args:
            hp (dict): Items of hyperparameter.

            necessary_items (list): List of necessary items.
                example: ["name", "type", "choices"]

            optional_items (list):  List of optional items.
                example: ["initial", "comment"]
        """
        for hp in self.hyperparameters:
            if hp['type'] == 'ordinal':
                if self.check_hyperparameters_item(
                    hp, necessary_items, optional_items
                ) is False:
                    sys.exit()

    def check_initial_type(self, types: list) -> None:
        """
            Check the format defalt value.
        Note:
            When Nelder-Mead is used, it can be defined
            in the form of a list, but in other cases,
            a list cannot be used.
        """
        for hp in self.hyperparameters:
            if 'initial' in hp.keys():
                if type(hp['initial']) not in types:
                    Terminal().print_error(
                        "default values tpye: '{}' "
                        "is not suportted in {}"
                        .format(type(hp['initial']), self.algorithm)
                    )
                    sys.exit()
