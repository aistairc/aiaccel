from __future__ import annotations

import copy
from typing import Any

from omegaconf.dictconfig import DictConfig

from aiaccel.config import is_multi_objective
from aiaccel.converted_parameter import ConvertedParameterConfiguration
from aiaccel.optimizer import AbstractOptimizer, NelderMead


class NelderMeadOptimizer(AbstractOptimizer):
    """An optimizer class with nelder mead algorithm.

    Args:
        config (DictConfig): A DictConfig object which contains optimization
            settings specified by the configuration file and the command line
            options.

    Attributes:
        nelder_mead (NelderMead): A class object implementing Nelder-Mead
            method.
        parameter_pool (list): A pool of parameters waiting for the process.
        order (list): A list of parameters being processed.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.params: ConvertedParameterConfiguration = ConvertedParameterConfiguration(
            self.params, convert_log=True, convert_int=True, convert_choices=True, convert_sequence=True
        )
        self.nelder_mead: Any = None
        self.parameter_pool: list[dict[str, Any]] = []
        self.order: list[int] = []

        if is_multi_objective(self.config):
            raise NotImplementedError("Nelder-Mead optimizer does not support multi-objective optimization.")

    def generate_initial_parameter(self) -> list[dict[str, float | int | str]] | None:
        """Generate initial parameters.

        Returns:
            list[dict[str, float | int | str]] | None: A list of new
            parameters. None if `self.nelder_mead` is already defined.
        """
        initial_parameter = super().generate_initial_parameter()
        if self.nelder_mead is not None:
            return None

        self.nelder_mead = NelderMead(
            self.params.get_parameter_list(), initial_parameters=initial_parameter, rng=self._rng
        )

        return self.generate_parameter()

    def check_result(self) -> None:
        pass

    def get_ready_parameters(self) -> list[Any]:
        """Get the list of ready parameters.

        Returns:
            list
        """
        return self.nelder_mead._executing

    def get_nm_results(self) -> list[dict[str, str | int | list[Any] | bool]]:
        """Get the list of Nelder-Mead result.

        Returns:
            list[dict[str, str | int | list | bool]]: Results per trial.
        """
        nm_results = []
        for p in self.get_ready_parameters():
            try:
                index = int(p["vertex_id"])
            except ValueError:
                continue
            except KeyError:
                continue

            result = self.get_any_trial_objective(index)

            if result is not None:
                nm_result = copy.copy(p)
                nm_result["result"] = result
                nm_results.append(nm_result)

        return nm_results

    def _add_result(self, nm_results: list[Any]) -> None:
        """Add a result parameter.

        Args:
            nm_results (list):

        Returns:
            None
        """
        if len(nm_results) == 0 or len(self.order) == 0:
            return

        # Store results in order of HP file generation
        order = self.order[0]
        for nm_result in nm_results:
            if order == nm_result["vertex_id"]:
                self.nelder_mead.add_result_parameters(nm_result)
                self.order.pop(0)
                break

    def update_ready_parameter_name(
        self, pool_p: dict[str, Any], name: Any  # old_param_name  # new_param_name
    ) -> None:
        """Update hyperparameter's names.

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
        old_param_name = pool_p["vertex_id"]
        new_param_name = name
        for e in self.nelder_mead._executing:
            if e["vertex_id"] == old_param_name:
                e["vertex_id"] = new_param_name
                break

    def nelder_mead_main(self) -> list[Any] | None:
        """Nelder Mead's main module.

        Args:
            None

        Returns:
            searched_params (list): Result of optimization.
        """
        searched_params = self.nelder_mead.search()
        if searched_params is None:
            self.logger.info("generate_parameter(): reached to max iteration.")
            return None
        if len(searched_params) == 0:
            return None
        return searched_params

    def _get_all_trial_id(self) -> list[Any]:
        """_get_all_trial_id.

        Get trial_ids from DB: 'result', 'finished', 'running', 'ready'

        Returns:
            list: trial_id
        """
        trial_id = self.storage.trial.get_all_trial_id()
        if trial_id is None:
            return []

        return trial_id

    def _get_current_names(self) -> list[Any]:
        """Get parameter trial_id.

        Returns:
            list: A list og parameter names in parameter_pool
        """
        # WARN: Always empty.
        return [p["vertex_id"] for p in self.parameter_pool]

    def generate_parameter(self) -> list[dict[str, float | int | str]] | None:
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
            if p["vertex_id"] not in self._get_all_trial_id() and p["vertex_id"] not in self._get_current_names():
                self.parameter_pool.append(copy.copy(p))

        if len(self.parameter_pool) == 0:
            return None

        pool_p = self.parameter_pool.pop(0)

        new_params = self.params.to_original_repr(pool_p["parameters"])

        self.update_ready_parameter_name(pool_p, self.trial_id.get())
        self.order.append(self.trial_id.get())

        return new_params
