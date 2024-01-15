"""Common variables and methods.

Example: ::

    from aiaccel.common import dict_lock

"""

dict_lock = "lock"
dict_log = "log"
dict_error = "error"
dict_stdout = "abci_stdout"
dict_stderr = "abci_stderr"
dict_runner = "runner"
dict_tensorboard = "tensorboard"
dict_mpi = "mpi"
dict_rank_log = "rank_log"

file_final_result = "final_result.yaml"
file_best_result = "best_result.yaml"
file_result_csv = "result.csv"
file_storage = "storage.db"

file_hp_count = "count.txt"
file_hp_count_lock = "count.lock"
file_hp_count_lock_timeout = 10

file_mpi_lock = "mpi.lock"
file_mpi_lock_timeout = 10

goal_maximize = "maximize"
goal_minimize = "minimize"

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
