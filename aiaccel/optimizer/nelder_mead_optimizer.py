from __future__ import annotations
import copy

from aiaccel.optimizer._nelder_mead import NelderMead
from aiaccel.optimizer.abstract_optimizer import AbstractOptimizer
from aiaccel.parameter import HyperParameter, HyperParameterConfiguration


class NelderMeadOptimizer(AbstractOptimizer):
    """An optimizer class with nelder mead algorithm.

    Args:
        options (dict[str, str | int | bool]): A dictionary containing
        command line options.

    Attributes:
        nelder_mead (NelderMead): A class object implementing Nelder-Mead
            method.
        parameter_pool (list): A pool of parameters waiting for the process.
        order (list): A list of parameters being processed.
    """

    def __init__(self, options: dict[str, str | int | bool]) -> None:
        super().__init__(options)
        self.nelder_mead = None
        self.parameter_pool = []
        self.order = []

    def generate_initial_parameter(
        self
    ) -> list[dict[str, float | int | str]] | None:
        """Generate initial parameters.

        Returns:
            list[dict[str, float | int | str]] | None: A list of new
            parameters. None if `self.nelder_mead` is already defined.
        """
        initial_parameter = super().generate_initial_parameter()
        if self.nelder_mead is not None:
            return

        self.params = self.special_settings_when_using_ordinal(self.params)

        self.nelder_mead = NelderMead(
            self.params.get_parameter_list(),
            initial_parameters=initial_parameter,
            rng=self._rng
        )

        return self.generate_parameter()

    def check_result(self) -> None:
        pass

    def get_ready_parameters(self) -> list:
        """Get the list of ready parameters.

        Returns:
            list
        """
        return self.nelder_mead._executing

    def get_nm_results(self) -> list[dict[str, str | int | list | bool]]:
        """ Get the list of Nelder-Mead result.

        Returns:
            list[dict[str, str | int | list | bool]]: Results per trial.
        """
        nm_results = []
        for p in self.get_ready_parameters():
            try:
                index = int(p['vertex_id'])
            except ValueError:
                continue
            except KeyError:
                continue

            result = self.storage.result.get_any_trial_objective(index)

            if result is not None:
                nm_result = copy.copy(p)
                nm_result['result'] = result
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
            - before::

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

            - after::

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
            self._logger.info('generate_parameter(): reached to max iteration.')
            return None
        if len(searched_params) == 0:
            return None
        return searched_params

    def _get_all_trial_id(self) -> list:
        """_get_all_trial_id.

        Get trial_ids from DB: 'result', 'finished', 'running', 'ready'

        Returns:
            list: trial_id
        """
        trial_id = self.storage.trial.get_all_trial_id()
        if trial_id is None:
            return []

        return trial_id

    def _get_current_names(self) -> list:
        """Get parameter trial_id.

        Returns:
            list: A list og parameter names in parameter_pool
        """
        # WARN: Always empty.
        return [p['vertex_id'] for p in self.parameter_pool]

    def generate_parameter(
        self
    ) -> list[dict[str, float | int | str]] | None:
        """Generate parameters.

        Returns:
            list[dict[str, float | int | str]] | None: A list of created
            parameters.

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

        new_params = []

        if len(self.parameter_pool) == 0:
            return new_params

        pool_p = self.parameter_pool.pop(0)

        for param in self.params.get_parameter_list():
            i = [p['parameter_name'] for p in pool_p['parameters']].index(param.name)
            if param.type.lower() == 'float':
                value = float(pool_p['parameters'][i]['value'])
            elif param.type.lower() == 'int':
                value = int(pool_p['parameters'][i]['value'])
            elif param.type.lower() == 'ordinal':
                index = int(pool_p['parameters'][i]['value'])
                value = param.sequence[index]
            else:
                raise TypeError(
                    'Invalid parameter type for NelderMeadSearch.'
                    f'FLOAT or INT is required, but {param.type} is given.'
                )

            new_params.append(
                {
                    'parameter_name': param.name,
                    'type': param.type,
                    'value': value
                }
            )

        self.update_ready_parameter_name(pool_p, self.trial_id.get())
        self.order.append(
            {
                'vertex_id': self.trial_id.get(),
                'parameters': new_params
            }
        )

        return new_params

    def special_settings_when_using_ordinal(self, params: HyperParameterConfiguration) -> HyperParameterConfiguration:
        """
            When using ordinal types in NelderMead, the array index is predicted.
            https://github.com/aistairc/aiaccel/issues/175
        """
        new_params = copy.deepcopy(params)
        for param in params.get_parameter_list():
            if param.type.lower() == 'ordinal':
                if param.name not in new_params.hps.keys():
                    assert False
                new_params.hps[param.name] = HyperParameter({
                    'name': param.name,
                    'type': 'ordinal',
                    'lower': 0,
                    'upper': len(param.sequence) - 1,
                    'sequence': param.sequence
                })

        return new_params
