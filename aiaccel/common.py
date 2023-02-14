"""Common variables and methods.

Example:

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
dict_verification = 'verification'
dict_storage = 'storage'

extension_hp = 'hp'
extension_pickle = 'pickle'
extension_resource = 'res'
extension_result = 'result'
extension_verification = 'verification'

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

search_algorithm_grid = 'aiaccel.optimizer.GridOptimizer'
search_algorithm_nelder_mead = 'aiaccel.optimizer.NelderMeadOptimizer'
search_algorithm_random = 'aiaccel.optimizer.RandomOptimizer'
search_algorithm_sobol = 'aiaccel.optimizer.SobolOptimizer'
search_algorithm_tpe = 'aiaccel.optimizer.TpeOptimizer'
