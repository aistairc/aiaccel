"""Common variables and methods.

Example: ::

    from aiaccel.common import alive_master

"""

alive_master = 'master.yml'
alive_optimizer = 'optimizer.yml'
alive_scheduler = 'scheduler.yml'

class_master = 'Master'
class_optimizer = 'Optimizer'
class_scheduler = 'Scheduler'

dict_work = 'work_aiaccel'
dict_alive = 'alive'
dict_pid = 'pid'
dict_ready = 'ready'
dict_running = 'running'
dict_finished = 'finished'
dict_hp = 'hp'
dict_hp_ready = 'hp/ready'
dict_hp_running = 'hp/running'
dict_hp_finished = 'hp/finished'
dict_srialize = 'serialize'
dict_lock = 'lock'
dict_log = 'log'
dict_error = 'error'
dict_output = 'abci_output'
dict_jobstate = 'jobstate'
dict_result = 'result'
dict_runner = 'runner'
dict_timestamp = 'timestamp'
dict_storage = 'storage'
dict_tensorboard = 'tensorboard'

extension_hp = 'hp'
extension_pickle = 'pickle'
extension_resource = 'res'
extension_result = 'result'

file_configspace = 'configspace'
file_final_result = 'final_result.result'
file_hyperparameter = 'hyperparameter.json'
file_numpy_random = 'numpy_random'
file_numpy_random_extension = 'npy'

file_hp_count = 'count.txt'
file_hp_count_lock = 'count.lock'
file_hp_count_lock_timeout = 10

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


class DataType:
    def is_uniform_int(self, data_type: str) -> bool:
        return data_type.lower() == data_type_uniform_int

    def is_uniform_float(self, data_type: str) -> bool:
        return data_type.lower() == data_type_uniform_float

    def is_categorical(self, data_type: str) -> bool:
        return data_type.lower() == data_type_categorical

    def is_ordinal(self, data_type: str) -> bool:
        return data_type.lower() == data_type_ordinal


data_type = DataType()
