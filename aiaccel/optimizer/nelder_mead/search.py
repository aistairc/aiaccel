from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer
from aiaccel.optimizer.nelder_mead.sampler import NelderMead
from aiaccel.util.filesystem import get_basename, get_file_hp_finished,\
    get_file_hp_ready, get_file_hp_running, get_file_result, load_yaml
from typing import Optional
import copy
import numpy as np


class NelderMeadSearchOptimizer(AbstractOptimizer):
    """An optimizer class with nelder mead algorithm.

    Attributes:
        nelder_mead ():
        parameter_pool ():
    """

    def __init__(self, options: dict) -> None:
        """Initial method of Optimizer.

        Args:
            config (str): A file name of a configuration.
        """
        super().__init__(options)
        self.nelder_mead = None
        self.parameter_pool = None
        self.order = []

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

    def check_result(self) -> None:
        """Check the result files and update nelder mead object.

        Returns:
            None
        """
        result_files = get_file_result(self.ws, self.dict_lock)
        hashnames = [get_basename(f) for f in result_files]
        params = self.get_ready_parameters()

        for p in params:
            try:
                index = hashnames.index(p['name'])
                # logging
                self.logger.debug(
                    "(nelder-mead)check_result:: p['name']: {}"
                    .format(p['name'])
                )
                # logging
                for i in range(len(result_files)):
                    self.logger.debug(
                        '(nelder-mead)check_result:: result_files[{}]: {}'
                        .format(i, result_files[i])
                    )
            except ValueError:
                continue

            result_file = result_files[index]
            result_content = load_yaml(result_file, self.dict_lock)

            # logging
            self.logger.debug(
                '(nelder-mead)check_result:: result_file: {}'
                .format(result_file)
            )

            # logging
            self.logger.debug(
                '(nelder-mead)check_result:: result_content: {}'
                .format(result_content)
            )

            nm_result = copy.copy(p)
            nm_result['result'] = result_content['result']
            self.nelder_mead.add_result_parameters(nm_result)

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
        result_files = get_file_result(self.ws, self.dict_lock)
        hashnames = [get_basename(f) for f in result_files]
        nm_results = []
        for p in self.get_ready_parameters():
            try:
                index = hashnames.index(p['name'])
            except ValueError:
                continue
            result_content = load_yaml(result_files[index], self.dict_lock)
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
            if order['name'] == nm_result['name']:
                self.logger.info(
                    "(nm) order: {}".format(order)
                )
                self.logger.info(
                    "(nm) nm_result: {}".format(nm_result)
                )
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
                'name': 'CMTrNe5P8a',
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
                'name': '000014',
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
        old_param_name = pool_p['name']
        new_param_name = name
        for e in self.nelder_mead._executing:
            if e['name'] == old_param_name:
                e['name'] = new_param_name
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
            self.logger.info(
                'generate_parameter(): reached to max iteration.'
            )
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
        for _ in range(number_of_generate):
            if len(self.parameter_pool) == 0:
                self.logger.info('All parameters in pool has been generated.')
                break

            self.new_params = []
            pool_p = self.parameter_pool.pop(0)

            self.logger.debug(
                'pool_p: {}'.format(pool_p)
            )

            for param in self.params.get_parameter_list():
                i = [p['parameter_name'] for p in pool_p['parameters']].index(
                    param.name
                )

                if param.type == 'FLOAT':
                    value = float(pool_p['parameters'][i]['value'])
                elif param.type == 'INT':
                    value = int(pool_p['parameters'][i]['value'])
                else:
                    raise TypeError(
                        'Invalid parameter type for NelderMeadSearch.'
                        'FLOAT or INT is required, but {} is given.'
                        .format(param.type)
                    )

                self.new_params.append(
                    {
                        'parameter_name': param.name,
                        'type': param.type,
                        'value': value
                    }
                )

            name = self.create_parameter_file({'parameters': self.new_params})
            self.update_ready_parameter_name(pool_p, name)
            self.order.append(
                {
                    'name': name,
                    'parameters': self.new_params
                }
            )

    def _get_all_hashnames(self) -> list:
        """_get_all_hashnames.

        Get hashname from dirs: 'result', 'finished', 'running', 'ready'

        Returns:
            List: hashname
        """
        result_files = get_file_result(self.ws, self.dict_lock)
        finished_files = get_file_hp_finished(self.ws, self.dict_lock)
        running_files = get_file_hp_running(self.ws, self.dict_lock)
        ready_files = get_file_hp_ready(self.ws, self.dict_lock)
        hashnames = [get_basename(f) for f in result_files]
        hashnames += [get_basename(f) for f in finished_files]
        hashnames += [get_basename(f) for f in running_files]
        hashnames += [get_basename(f) for f in ready_files]
        return hashnames

    def _get_current_names(self):
        """ get parameter names.

        Returns:
            parameter names in parameter_pool (list)
        """
        # WARN: Always empty.
        return [p['name'] for p in self.parameter_pool]

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

        nm_results = self.get_nm_results()
        self._add_result(nm_results)

        searched_params = self.nelder_mead_main()

        self.logger.debug(
            "searched_params: {}".format(searched_params)
        )

        if searched_params is None:
            return None

        for p in searched_params:
            if (
                p['name'] not in self._get_all_hashnames() and
                p['name'] not in self._get_current_names()
            ):
                self.logger.debug(
                    "parameter_pool.append({})".format(p)
                )
                self.parameter_pool.append(copy.copy(p))

        self._generate_hp_ready(number)

    def _serialize(self) -> dict:
        """Serialize this module.

        Returns:
            dict: The serialized objects.
        """
        parameter_pool = copy.deepcopy(self.parameter_pool)
        for p_pool in parameter_pool:
            for p_pool_param in p_pool['parameters']:
                if type(p_pool_param['value']) is np.float64:
                    p_pool_param['value'] = float(p_pool_param['value'])

        self.serialize_datas = {
            'generated_parameter': self.generated_parameter,
            'loop_count': self.loop_count,
            'parameter_pool': parameter_pool,
            'nelder_mead': self.nelder_mead.serialize(),
            'order': self.order
        }
        return super()._serialize()

    def _deserialize(self, dict_objects: dict) -> None:
        """Deserialize this module.

        Args:
            dict_objects(dict): A dictionary including serialized objects.

        Returns:
            None
        """
        super()._deserialize(dict_objects)
        parameter_pool = copy.deepcopy(dict_objects['parameter_pool'])
        for p_pool in parameter_pool:
            for p_pool_param in p_pool['parameters']:
                if type(p_pool_param['value']) is float:
                    p_pool_param['value'] = np.float64(p_pool_param['value'])

        self.parameter_pool = parameter_pool
        self.nelder_mead = NelderMead(self.params.get_parameter_list())
        self.nelder_mead.deserialize(dict_objects['nelder_mead'])
        self.order = dict_objects['order']
