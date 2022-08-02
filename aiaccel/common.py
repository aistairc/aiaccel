from pathlib import Path

"""Common variables and methods.

    * Import this as follows:
"""
alive_master = 'master.yml'
alive_optimizer = 'optimizer.yml'
alive_scheduler = 'scheduler.yml'

class_master = 'Master'
class_optimizer = 'Optimizer'
class_scheduler = 'Scheduler'

dict_work = 'work_aiaccel'
dict_alive = 'alive'
dict_hp = 'hp'
dict_hp_ready = 'hp/ready'
dict_hp_running = 'hp/running'
dict_hp_finished = 'hp/finished'
dict_lock = 'lock'
dict_log = 'log'
dict_output = 'abci_output'
dict_state = 'state'
dict_result = 'result'
dict_runner = 'runner'
dict_verification = 'verification'

extension_hp = 'hp'
extension_pickle = 'pickle'
extension_resource = 'res'
extension_result = 'result'
extension_verification = 'verification'

file_configspace = 'configspace'
file_final_result = 'final_result.result'
file_hyperparameter = 'hyperparameter.json'
file_native_random = 'native_random'
file_numpy_random = 'numpy_random'
file_numpy_random_extension = 'npy'

file_hp_count = 'count.txt'  # wd/
file_hp_count_lock = 'count.lock'  # wd/
file_hp_count_lock_timeout = 10  # wd/

goal_maximize = 'maximize'
goal_minimize = 'minimize'

key_module_type = 'module_type'
key_path = 'path'
key_pid = 'pid'
key_project_name = 'project_name'

module_type_master = 'master'
module_type_optimizer = 'optimizer'
module_type_scheduler = 'scheduler'

resource_type_local = 'local'
resource_type_abci = 'abci'

search_algorithm_grid = 'aiaccel.optimizer.grid'
search_algorithm_nelder_mead = 'aiaccel.optimizer.nelder_mead'
search_algorithm_random = 'aiaccel.optimizer.random'
search_algorithm_sobol = 'aiaccel.optimizer.sobol'
search_algorithm_tpe = 'aiaccel.optimizer.tpe'


def get_module_type_from_class_name(class_name: str) -> str:
    """Get a module name(master, optimizer or scheduler) from class name.

    Args:
        class_name(str): A class name of caller.

    Returns:
        str: A module name string "master", "optimizer" or "scheduler".

    Raises:
        TypeError: if class_name does not match module names.

    """
    if class_master in class_name:
        return module_type_master
    elif class_optimizer in class_name:
        return module_type_optimizer
    elif class_scheduler in class_name:
        return module_type_scheduler
    else:
        raise TypeError('A class name does not match module names.')


def get_file_random(
    path: Path,
    class_name: str,
    loop_count: int,
    random_type: str
) -> Path:
    """Get a file path to serialize a random generator.

    Args:
        path (Path): A path to a state directory.
        class_name (str): A class name of a caller module.
        loop_count (int): A loop count.
        random_type (str): A random type in native or numpy.

    Returns:
        Path: A path to serialize a specified random generator.
    Raises:
        TypeError: if invalid random_type is set.
    """
    class_str = get_module_type_from_class_name(class_name)

    random_types = [
        file_configspace,
        file_native_random,
        file_numpy_random
    ]

    if random_type not in random_types:
        raise TypeError

    if random_type is file_configspace:
        return path.joinpath(
            '{}_{}_{:010}.{}'
            .format(
                class_str,
                random_type,
                loop_count,
                extension_pickle
            )
        )
    elif random_type is file_native_random:
        return path.joinpath(
            '{}_{}_{:010}.{}'
            .format(
                class_str,
                random_type,
                loop_count,
                extension_pickle
            )
        )
    elif random_type is file_numpy_random:
        return path.joinpath(
            '{}_{}_{:010}.{}'
            .format(
                class_str,
                random_type,
                loop_count,
                extension_pickle
            )
        )
