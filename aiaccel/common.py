"""Common variables and methods.

Example: ::

    from aiaccel.common import alive_master

"""

class_master = 'Master'
class_optimizer = 'Optimizer'
class_scheduler = 'Scheduler'

dict_hp = 'hp'
dict_lock = 'lock'
dict_log = 'log'
dict_error = 'error'
dict_output = 'abci_output'
dict_runner = 'runner'
dict_verification = 'verification'
dict_storage = 'storage'

extension_verification = 'verification'

file_hp_count = 'count.txt'
file_hp_count_lock = 'count.lock'
file_hp_count_lock_timeout = 10
file_final_result = 'final_result.result'

goal_maximize = 'maximize'
goal_minimize = 'minimize'

module_type_master = 'master'
module_type_optimizer = 'optimizer'
module_type_scheduler = 'scheduler'

resource_type_local = 'local'
resource_type_abci = 'abci'
resource_type_python_local = 'python_local'

data_type_uniform_int = 'uniform_int'
data_type_uniform_float = 'uniform_float'
data_type_categorical = 'categorical'
data_type_ordinal = 'ordinal'

search_algorithm_budget_specified_grid = 'aiaccel.optimizer.BudgetSpecifiedGridOptimizer'
search_algorithm_grid = 'aiaccel.optimizer.GridOptimizer'
search_algorithm_nelder_mead = 'aiaccel.optimizer.NelderMeadOptimizer'
search_algorithm_random = 'aiaccel.optimizer.RandomOptimizer'
search_algorithm_sobol = 'aiaccel.optimizer.SobolOptimizer'
search_algorithm_tpe = 'aiaccel.optimizer.TpeOptimizer'
