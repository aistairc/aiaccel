from aiaccel.optimizer.abstract import AbstractOptimizer
from aiaccel.optimizer.nelder_mead.sampler import NelderMead
from typing import Optional
import copy
import numpy as np


class NelderMeadOptimizer(AbstractOptimizer):
    """An optimizer class with nelder mead algorithm.

    Attributes:
        nelder_mead ():
        parameter_pool ():
    """

    def __init__(self, options: dict) -> None:
        """Initial method of NelderMeadOptimizer.

        Args:
            config (str): A file name of a configuration.
        """
        super().__init__(options)
        self.nelder_mead = None
        self.parameter_pool = None
        self.order = []

        self.storage.variable.register(
            process_name=self.options['process_name'],
            labels=[
                'native_random_state',
                'numpy_random_state',
                'num_of_generated_parameter',
                'loop_count',
                'parameter_pool',
                'nelder_mead',
                'order'
            ]
        )

    def pre_process(self) -> None:
        """Pre-procedure before executing processes.

        Returns:
            None
        """
        super().pre_process()


        initial_parameter = self.generate_initial_parameter()
        if self.nelder_mead is not None:
            return

        if initial_parameter is not None:
            self.nelder_mead = NelderMead(
                self.params.get_parameter_list(),
                initial_parameters=initial_parameter['parameters']
            )
        else:
            self.nelder_mead = NelderMead(self.params.get_parameter_list())
        self.parameter_pool = []

    def set_maximize(self):
        """ Set the goal to Maximize.

        Returns:
            None
        """
        self.nelder_mead._maximize = True

    def set_minimize(self):
        """ Set the goal to Minimize.

        Returns:
            None
        """
        self.nelder_mead._maximize = False

    # def check_result(self) -> None:
    #     """Check the result files and update nelder mead object.

    #     Returns:
    #         None
    #     """
    #     # result_files = get_file_result(self.ws, self.dict_lock)
    #     results = self.storage.result.get_result_trial_id_list()
    #     params = self.get_ready_parameters()
    #     for p in params:
    #         try:
    #             int(p['name'])
    #         except ValueError:
    #             continue

    #         if self.storage.is_finished(trial_id=int(p['name'])):
    #             index = p['name']
    #             self.logger.debug(f"(nelder-mead)check_result:: p['name']: {p['name']}")
    #             for i in range(len(results)):
    #                 self.logger.debug(f'(nelder-mead)check_result:: trial[{i}]: {results[i]}')
    #         else:
    #             continue

    #         nm_result = self.storage.get_hp_dict(trial_id_str=index)
    #         self.logger.debug(f"nm_result :{nm_result}")
    #         self.nelder_mead.add_result_parameters(nm_result)

    def check_result(self) -> None:
        pass

    def get_ready_parameters(self) -> list:
        """ Get the list of ready parameters.

        Returns:
            list
        """
        return self.nelder_mead._executing

    def get_nm_results(self) -> list:
        """ Get the list of Nelder-Mead result.

        Returns:
            list[dict]: Results per trial.
        """
        results = self.storage.result.get_result_trial_id_list()
        nm_results = []
        for p in self.get_ready_parameters():
            try:
                int(p['vertex_id'])
            except ValueError:
                continue
            except KeyError:
                continue

            if int(p['vertex_id']) in results:
                index = p['vertex_id']
            else:
                continue

            result_content = self.storage.get_hp_dict(trial_id_str=index)
            nm_result = copy.copy(p)
            nm_result['result'] = result_content['result']
            nm_results.append(nm_result)
        return nm_results

    def _add_result(self, nm_results: list) -> None:
        """  Add a result parameter.

        Args:
            nm_results (list):

        Returns:
            None
        """
        if (
            len(nm_results) == 0 or
            len(self.order) == 0
        ):
            return

        # Store results in order of HP file generation
        order = self.order[0]
        for nm_result in nm_results:
            if order['vertex_id'] == nm_result['vertex_id']:
                self.nelder_mead.add_result_parameters(nm_result)
                self.order.pop(0)
                break

    def update_ready_parameter_name(
        self,
        pool_p: str,  # old_param_name
        name: str     # new_param_name
    ) -> None:
        """ Update hyperparameter's names.

        Args:
            pool_p (str): old parameter name
            name (str): New parameter name

        Returns:
            None

        Note:
            --------------------------------------------------------------
            - befor
            {
                'vertex_id': 'CMTrNe5P8a',
                'parameters': [
                    {'parameter_name': 'x1', 'value': 3.37640289751353},
                    {'parameter_name': 'x2', 'value': 1.6556037243290205}
                ],
                'state': 'WaitExpand',
                'itr': 5,
                'index': None,
                'out_of_boundary': False
            }
            --------------------------------------------------------------
            - after
            {
                'vertex_id': '000014', <---- replace to trial_id
                'parameters': [
                    {'parameter_name': 'x1', 'value': 3.37640289751353},
                    {'parameter_name': 'x2', 'value': 1.6556037243290205}
                ],
                'state': 'WaitExpand',
                'itr': 5,
                'index': None,
                'out_of_boundary': False
            }
            --------------------------------------------------------------
        """
        old_param_name = pool_p['vertex_id']
        new_param_name = name
        for e in self.nelder_mead._executing:
            if e['vertex_id'] == old_param_name:
                e['vertex_id'] = new_param_name
                break

    def nelder_mead_main(self) -> list:
        """ Nelder Mead's main module.

        Args:
            None

        Returns:
            searched_params (list): Result of optimization.
        """
        searched_params = self.nelder_mead.search()
        if searched_params is None:
            self.logger.info('generate_parameter(): reached to max iteration.')
            return None
        if len(searched_params) == 0:
            return None
        return searched_params

    def _generate_hp_ready(self, number_of_generate: int) -> None:
        """ generate hp-file for ready.

        The generated parameter data will be append to 'self.order'

        Args:
            number_of_generate (int): The number of generate parameter.

        Returns:
            None
        """
        trial_id = self.trial_id.integer
        for _ in range(number_of_generate):
            if len(self.parameter_pool) == 0:
                self.logger.info('All parameters in pool has been generated.')
                break

            self.new_params = []
            pool_p = self.parameter_pool.pop(0)

            for param in self.params.get_parameter_list():
                i = [p['parameter_name'] for p in pool_p['parameters']].index(param.name)

                if param.type == 'FLOAT':
                    value = float(pool_p['parameters'][i]['value'])
                elif param.type == 'INT':
                    value = int(pool_p['parameters'][i]['value'])
                else:
                    raise TypeError(
                        'Invalid parameter type for NelderMeadSearch.'
                        f'FLOAT or INT is required, but {param.type} is given.'
                    )

                self.new_params.append(
                    {
                        'parameter_name': param.name,
                        'type': param.type,
                        'value': value
                    }
                )

            self.register_ready({'parameters': self.new_params})
            self.update_ready_parameter_name(pool_p, self.trial_id.get())
            self.order.append(
                {
                    'vertex_id': self.trial_id.get(),
                    'parameters': self.new_params
                }
            )
            self._serialize(trial_id)

    def _get_all_trial_id(self) -> list:
        """_get_all_trial_id.

        Get trial_ids from DB: 'result', 'finished', 'running', 'ready'

        Returns:
            List: trial_id
        """

        trial_id = self.storage.trial.get_all_trial_id()
        if trial_id is None:
            return []

        return trial_id

    def _get_current_names(self):
        """ get parameter trial_id.

        Returns:
            parameter names in parameter_pool (list)
        """
        # WARN: Always empty.
        return [p['vertex_id'] for p in self.parameter_pool]

    def generate_parameter(
        self,
        number: Optional[int] = 1  # A number of generating parameters.
    ) -> None:
        """Generate parameters.
        Args:
            number (Optional[int]):
                A number of generating parameters.

        Returns:
            None

        Raises:
            TypeError: Causes when an invalid parameter type is set.
        """

        self._add_result(self.get_nm_results())

        searched_params = self.nelder_mead_main()

        if searched_params is None:
            return None

        for p in searched_params:
            if (
                p['vertex_id'] not in self._get_all_trial_id() and
                p['vertex_id'] not in self._get_current_names()
            ):
                self.parameter_pool.append(copy.copy(p))

        self._generate_hp_ready(number)

    def _serialize(self, trial_id) -> None:
        """Serialize this module.

        Returns:
            dict: The serialized objects.
        """

        parameter_pool = copy.deepcopy(self.parameter_pool)
        for p_pool in parameter_pool:
            for p_pool_param in p_pool['parameters']:
                if type(p_pool_param['value']) is np.float64:
                    p_pool_param['value'] = float(p_pool_param['value'])

        self.storage.variable.d['native_random_state'].set(trial_id=trial_id, value=self.get_native_random_state())
        self.storage.variable.d['numpy_random_state'].set(trial_id=trial_id, value=self.get_numpy_random_state())
        self.storage.variable.d['num_of_generated_parameter'].set(trial_id=trial_id, value=self.num_of_generated_parameter)
        self.storage.variable.d['loop_count'].set(trial_id=trial_id, value=self.loop_count)
        self.storage.variable.d['parameter_pool'].set(trial_id=trial_id, value=parameter_pool)
        self.storage.variable.d['nelder_mead'].set(trial_id=trial_id, value=self.nelder_mead.serialize())
        self.storage.variable.d['order'].set(trial_id=trial_id, value=self.order)

    def _deserialize(self, trial_id: int) -> None:
        """Deserialize this module.

        Args:
            dict_objects(dict): A dictionary including serialized objects.

        Returns:
            None
        """
        self.set_native_random_state(self.storage.variable.d['native_random_state'].get(trial_id))
        self.set_numpy_random_state(self.storage.variable.d['numpy_random_state'].get(trial_id))

        self.num_of_generated_parameter = self.storage.variable.d['num_of_generated_parameter'].get(trial_id)
        self.loop_count = self.storage.variable.d['loop_count'].get(trial_id)

        parameter_pool = copy.deepcopy(self.storage.variable.d['parameter_pool'].get(trial_id))
        for p_pool in parameter_pool:
            for p_pool_param in p_pool['parameters']:
                if type(p_pool_param['value']) is float:
                    p_pool_param['value'] = np.float64(p_pool_param['value'])
        self.parameter_pool = parameter_pool

        self.nelder_mead = NelderMead(self.params.get_parameter_list())
        self.nelder_mead.deserialize(self.storage.variable.d['nelder_mead'].get(trial_id))

        self.order = self.storage.variable.d['order'].get(trial_id)
