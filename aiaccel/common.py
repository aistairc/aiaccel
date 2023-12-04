"""Common variables and methods.

Example: ::

    from aiaccel.common import alive_optimizer

"""

alive_optimizer = "optimizer.yml"
alive_manager = "manager.yml"

class_optimizer = "Optimizer"
class_manager = "Manager"

dict_work = "work_aiaccel"
dict_alive = "alive"
dict_pid = "pid"
dict_ready = "ready"
dict_running = "running"
dict_finished = "finished"
dict_hp = "hp"
dict_hp_ready = "hp/ready"
dict_hp_running = "hp/running"
dict_hp_finished = "hp/finished"
dict_srialize = "serialize"
dict_lock = "lock"
dict_log = "log"
dict_error = "error"
dict_output = "abci_output"
dict_jobstate = "jobstate"
dict_result = "result"
dict_runner = "runner"
dict_timestamp = "timestamp"
dict_storage = "storage"
dict_tensorboard = "tensorboard"
dict_mpi = "mpi"
dict_rank_log = "rank_log"

extension_hp = "hp"
extension_pickle = "pickle"
extension_resource = "res"
extension_result = "result"

file_configspace = "configspace"
file_final_result = "final_result.result"
file_hyperparameter = "hyperparameter.json"
file_numpy_random = "numpy_random"
file_numpy_random_extension = "npy"

file_hp_count = "count.txt"
file_hp_count_lock = "count.lock"
file_hp_count_lock_timeout = 10

file_mpi_lock = "mpi.lock"
file_mpi_lock_timeout = 10

goal_maximize = "maximize"
goal_minimize = "minimize"

key_module_type = "module_type"
key_path = "path"
key_pid = "pid"
key_project_name = "project_name"

resource_type_local = "local"
resource_type_abci = "abci"
resource_type_mpi = "mpi"
resource_type_python_local = "python_local"

search_algorithm_budget_specified_grid = "aiaccel.optimizer.BudgetSpecifiedGridOptimizer"
search_algorithm_grid = "aiaccel.optimizer.GridOptimizer"
search_algorithm_nelder_mead = "aiaccel.optimizer.NelderMeadOptimizer"
search_algorithm_random = "aiaccel.optimizer.RandomOptimizer"
search_algorithm_sobol = "aiaccel.optimizer.SobolOptimizer"
search_algorithm_tpe = "aiaccel.optimizer.TpeOptimizer"

datetime_format = "%m/%d/%Y %H:%M:%S"
