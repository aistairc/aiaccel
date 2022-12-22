from typing import Union
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pathlib import Path


default_config = {
    'generic': {
        'workspace': './work',
        'job_command': '',
        'python_file': '',
        'function': '',
        'batch_job_timeout': 600,
        'sleep_time': 0.01
    },
    'resource': {
        'type': 'local',
        'num_node': 1
    },
    'ABCI': {
        'group': '[group]',
        'job_script_preamble': './job_script_preamble.sh',
        'job_execution_options': '',
        'runner_search_pattern': ''
    },
    'optimize': {
        'search_algorithm': 'aiaccel.optimizer.NelderMeadOptimizer',
        'goal': 'minimize',
        'trial_number': 30,
        'rand_seed': 42,
        'sobol_scramble': True,
        'parameters': []
    },
    'job_setting': {
        'cancel_retry': 3,
        'cancel_timeout': 60,
        'expire_retry': 3,
        'expire_timeout': 60,
        'finished_retry': 3,
        'finished_timeout': 60,
        'job_retry': 2,
        'job_timeout': 60,
        'kill_retry': 3,
        'kill_timeout': 60,
        'result_retry': 1,
        'runner_retry': 3,
        'runner_timeout': 60,
        'running_retry': 3,
        'running_timeout': 60,
        'init_fail_count': 100,
        'name_length': 6,
        'random_scheduling': False
    },
    'logger': {
        'file': {
            'master': 'master.log',
            'optimizer': 'optimizer.log',
            'scheduler': 'scheduler.log'
        },
        'log_level': {
            'master': 'DEBUG',
            'optimizer': 'DEBUG',
            'scheduler': 'DEBUG'
        },
        'stream_level': {
            'master': 'DEBUG',
            'optimizer': 'DEBUG',
            'scheduler': 'DEBUG'
        }
    },
    'verification': {
        'is_verified': False,
        'condition': []
    }
}


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

    default = OmegaConf.create(default_config)
    customize = OmegaConf.load(path)

    config = OmegaConf.merge(default, customize)
    config.config_path = str(path)

    return config
