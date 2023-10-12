from __future__ import annotations

import copy
from typing import Any

import numpy as np
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig

from aiaccel.common import goal_maximize
from aiaccel.config import is_multi_objective
from aiaccel.converted_parameter import ConvertedParameterConfiguration
from aiaccel.optimizer import AbstractOptimizer
from aiaccel.optimizer._nelder_mead import NelderMead, Vertex
from aiaccel.optimizer.value import Value
from aiaccel.parameter import OrdinalParameter


class NelderMeadOptimizer(AbstractOptimizer):
    """An optimizer class with nelder mead algorithm.

    Args:
        config (DictConfig): A DictConfig object which contains optimization
            settings specified by the configuration file and the command line
            options.

    Attributes:
        nelder_mead (NelderMead): A class object implementing Nelder-Mead
            method.
    """

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.params: ConvertedParameterConfiguration = ConvertedParameterConfiguration(
            self.params, convert_log=True, convert_int=True, convert_choices=True, convert_sequence=True
        )
        self.base_params = self.params.get_empty_parameter_dict()
        self.n_params = len(self.params.get_parameter_list())
        self.param_names = self.params.get_parameter_names()
        self.bdrys = np.array([[p.lower, p.upper] for p in self.params.get_parameter_list()])
        self.n_dim = len(self.bdrys)
        self.nelder_mead: Any = None
        if is_multi_objective(self.config):
            raise NotImplementedError("Nelder-Mead optimizer does not support multi-objective optimization.")
        self.single_or_multiple_trial_params: list[Vertex] = []
        self.map_trial_id_and_vertex_id: dict[int, str] = {}
        self.completed_trial_ids: list[int] = []

    def convert_ndarray_to_parameter(self, ndarray: np.ndarray[Any, Any]) -> list[dict[str, float | int | str]]:
        """Convert a list of numpy.ndarray to a list of parameters."""
        new_params = copy.deepcopy(self.base_params)
        for name, value, b in zip(self.param_names, ndarray, self.bdrys):
            for new_param in new_params:
                if new_param["parameter_name"] == name:
                    new_param["value"] = value
                if b[0] <= value <= b[1]:
                    new_param["out_of_boundary"] = False
                else:
                    new_param["out_of_boundary"] = True
        return new_params

    def new_finished(self) -> list[int]:
        finished = self.storage.get_finished()
        return list(set(finished) ^ set(self.completed_trial_ids))

    def _generate_initial_parameter(self, initial_parameters: Any, dim: int, num_of_initials: int) -> Any:
        params = self.params.get_parameter_list()
        if initial_parameters is None:
            if isinstance(params[dim], OrdinalParameter):
                return self._rng.randint(len(params[dim].sequence))
            return params[dim].sample(rng=self._rng)["value"]

        if not isinstance(initial_parameters[dim]["value"], (list, ListConfig)):
            initial_parameters[dim]["value"] = [initial_parameters[dim]["value"]]

        if num_of_initials < len(initial_parameters[dim]["value"]):
            val = initial_parameters[dim]["value"][num_of_initials]
            return val

        else:
            val = params[dim].sample(rng=self._rng)["value"]
            return val

    def generate_initial_parameter(self) -> list[dict[str, float | int | str]] | None:
        """Generate initial parameters.

        Returns:
            list[dict[str, float | int | str]] | None: A list of new
            parameters. None if `self.nelder_mead` is already defined.
        """
        _initial_parameters = super().generate_initial_parameter()
        initial_parameters = np.array(
            [
                [
                    self._generate_initial_parameter(_initial_parameters, dim, num_of_initials)
                    for dim in range(self.n_params)
                ]
                for num_of_initials in range(self.n_dim + 1)
            ]
        )

        self.logger.debug(f"initial_parameters: {initial_parameters}")
        if self.nelder_mead is not None:
            return None
        self.nelder_mead = NelderMead(initial_parameters=initial_parameters)
        return self.generate_parameter()

    def generate_parameter(self) -> list[dict[str, float | int | str]] | None:
        """Generate parameters.

        Returns:
            list[dict[str, float | int | str]] | None: A list of created
            parameters.

        Raises:
            TypeError: Causes when an invalid parameter type is set.
        """
        searched_params: list[Vertex] = self.nelder_mead_main()
        for searched_param in searched_params:
            self.single_or_multiple_trial_params.append(searched_param)
        if len(self.single_or_multiple_trial_params) == 0:
            return None
        new_params: Vertex = self.single_or_multiple_trial_params.pop(0)
        self.map_trial_id_and_vertex_id[self.trial_id.integer] = new_params.id
        new_param = self.convert_ndarray_to_parameter(new_params.coordinates)
        return new_param

    def inner_loop_main_process(self) -> bool:
        """A main loop process. This process is repeated every main loop.

        Returns:
            bool: The process succeeds or not. The main loop exits if failed.
        """
        self.update_each_state_count()
        if self.check_finished():
            return False
        if self.all_parameters_processed():
            return False
        if self.all_parameters_registered():
            return True
        pool_size = self.get_pool_size()
        if pool_size == 0:
            return True
        self.logger.info(
            f"hp_ready: {self.hp_ready}, "
            f"hp_running: {self.hp_running}, "
            f"hp_finished: {self.hp_finished}, "
            f"total: {self.config.optimize.trial_number}, "
            f"pool_size: {pool_size}"
        )
        if new_params := self.generate_new_parameter():
            if self.out_of_boundary(new_params):
                self.logger.debug(f"out of boundary: {new_params}")
                self.register_new_parameters(self.convert_type_by_config(new_params), state="finished")
                objective = np.inf
                if self.goals[0] == goal_maximize:
                    objective = -np.inf
                self.storage.result.set_any_trial_objective(trial_id=self.trial_id.integer, objective=[objective])
                self.trial_id.increment()
                self._serialize(self.trial_id.integer)
                return True
            self.register_new_parameters(self.convert_type_by_config(new_params))
            self.trial_id.increment()
            self._serialize(self.trial_id.integer)
            return True
        self.print_dict_state()

        return True

    def out_of_boundary(self, params: list[dict[str, float | int | str]]) -> bool:
        for param in params:
            if param["out_of_boundary"]:
                return True
        return False

    def nelder_mead_main(self) -> list[Vertex]:
        """Nelder Mead's main module.

        Args:
            None

        Returns:
            searched_params (list): Result of optimization.
        """

        nm_state = self.nelder_mead.get_state()
        if nm_state in {
            "initialize_pending",
            "reflect_pending",
            "expand_pending",
            "inside_contract_pending",
            "outside_contract_pending",
            "shrink_pending",
        }:
            new_finished = self.new_finished()
            if len(new_finished) == self.nelder_mead.get_n_waits():
                if len(new_finished) > 1:
                    values = []
                    for trial_id in new_finished:
                        self.completed_trial_ids.append(trial_id)
                        vertex_id = self.map_trial_id_and_vertex_id[trial_id]
                        objective = self.storage.result.get_any_trial_objective(trial_id)[0]
                        if self.goals[0] == goal_maximize:
                            objective *= -1.0
                        values.append(Value(id=vertex_id, value=objective))
                    if nm_state == "initialize_pending":
                        self.nelder_mead.after_initialize(values)
                    elif nm_state == "shrink_pending":
                        self.nelder_mead.aftter_shrink(values)
                elif len(new_finished) == 1:
                    trial_id = new_finished[-1]
                    vertex_id = self.map_trial_id_and_vertex_id[trial_id]
                    objective = self.storage.result.get_any_trial_objective(trial_id)[0]
                    if self.goals[0] == goal_maximize:
                        objective *= -1.0
                    value = Value(id=vertex_id, value=objective)
                    self.completed_trial_ids.append(trial_id)
                    if nm_state == "reflect_pending":
                        self.nelder_mead.after_reflect(value)
                    elif nm_state == "expand_pending":
                        self.nelder_mead.after_expand(value)
                    elif nm_state == "inside_contract_pending":
                        self.nelder_mead.after_inside_contract(value)
                    elif nm_state == "outside_contract_pending":
                        self.nelder_mead.after_outside_contract(value)
        elif nm_state == "initialize":
            ...
        elif nm_state == "reflect":
            ...
        elif nm_state == "expand":
            ...
        elif nm_state == "inside_contract":
            ...
        elif nm_state == "outside_contract":
            ...
        elif nm_state == "shrink":
            ...
        else:
            raise NotImplementedError(f"Invalid state: {nm_state}")  # not reachable
        searched_params = self.nelder_mead.search()
        return searched_params
