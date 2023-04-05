from __future__ import annotations

import logging
from typing import Any

import numpy as np
from omegaconf.listconfig import ListConfig

from aiaccel.parameter import HyperParameter
from aiaccel.util import generate_random_name

STATES = [
    'WaitInitialize',
    'Initialize',
    'WaitReflect',
    'ReflectBranch',
    'WaitExpand',
    'ExpandBranch',
    'WaitOutsideContract',
    'OutsideContractBranch',
    'WaitInsideContract',
    'InsideContractBranch',
    'WaitShrink'
]


class NelderMead(object):
    """A class implementing Nelder-Mead method.

    Args:
        params (list[HyperParameter]): A list of hyper parameter objects.
        iteration (float | None, optional): A max iteration counts.
            Defaults to float('inf').
        coef (dict | None, optional): A coefficient values. Defaults to None.
        maximize (bool | None, optional): Evaluate maximize or not. Defaults
            to False.
        initial_parameters (list[dict[str, str  |  float  |  list[float]]] | None, optional):
            A initial parameters. Defaults to None.
        rng (np.random.RandomState | None, optional): A reference to a random
            generator. Defaults to None.

    Attributes:
        bdrys (np.ndarray): A list of boundaries.
        coef (dict[str, float]): A dictionary of coefficients.
        f (np.ndarray): A list of evaluated parameter results.
        logger (logging.Logger): A logger object.
        params (list[HyperParameter]): A list of hyper parameters.
        storage (dict[str, float | None]): A dictionary to store temporal
            calculation results.
        y (np.ndarray): A list of current evaluated parameters.
        yc (float): A current centroid value of y.
        _evaluated_itr (int): A count of evaluation each loop.
        _executing (list[dict]): A list to store candidates to be executed.
        _executing_index (int): A number to be added to executing list.
        _fe (float): A temporal result of Expand.
        _fic (float): A temporal result of Inside Contraction.
        _foc (float): A temporal result of Outside Contraction.
        _fr (float): A temporal result of Reflection
        _history (dict[str, list[float | str]]): A storage of execution
            history of each value and operator.
        _maximize (bool): Evaluate the result as maximize or minimize.
        _max_itr (int): A number of max iterations. This is compared with
            _evaluated_itr.
        _num_shrink (int): A number of Shrink.
        _out_of_boundary (bool): Is a current iteration out of boundaries or
            not.
        _result (list[float]): A list of results for _executing.
        _state (str): A current state.
        _total_itr (int): A number of iterations. Currently same with
            _evaluated_itr. It's different if counts out of boundaries.

    Todo:
        Fix float comparison errors.
    """

    def __init__(
        self,
        params: list[HyperParameter],
        iteration: float = float('inf'),
        coef: dict[str, Any] | None = None,
        maximize: bool | None = False,
        initial_parameters: Any = None,
        rng: np.random.RandomState | None = None
    ) -> None:
        if coef is None:
            coef = {"r": 1.0, "ic": - 0.5, "oc": 0.5, "e": 2.0, "s": 0.5}

        self.logger = logging.getLogger('root.optimizer.nelder_mead')
        self.params = params
        self.bdrys = np.array([[p.lower, p.upper] for p in self.params])
        self.coef = coef
        self.n_dim = len(self.bdrys)
        self._rng = rng

        self.y = self._create_initial_values(initial_parameters)

        self.f: Any = []  # np.ndarray is assigned in self._wait_initialize().
        self.yc = None
        self._storage = {
            "r": None, "ic": None, "oc": None, "e": None, "s": None}
        self._max_itr = iteration
        self._total_itr = 0         # included out of boundary
        self._evaluated_itr = 0     # not included out of boundary
        self._history: dict[str, list[Any]] = {
            'total_y': [],              # y each loop
            'evaluated_y': [],          # y each loop not included out of boundary
            'op': [],                   # operations such as 'reflect' and so on.
            'total_sample': [],         # sampled point each loop
            'evaluated_sample': [],     # sampled point each loop not included out of boundary
            'fyr_order': []             # order of f(yr) in self.f
        }
        self._executing_index = 0
        self._executing: list[Any] = []
        self._fr: Any = None  # treated as "float" until self._finalize() where "None" is assigned.
        self._fe: Any = None
        self._fic: Any = None
        self._foc: Any = None
        self._maximize = maximize
        self._num_shrink = 0
        self._state = 'WaitInitialize'
        self._out_of_boundary = False
        self._result: list[Any] = []
        for y in self.y:
            self._add_executing(y)

    def _create_initial_values(
        self,
        initial_parameters: list[dict[str, Any]]
    ) -> np.ndarray[Any, Any]:
        initial_values = [
            [self._create_initial_value(initial_parameters, dim, num_of_initials) for dim in range(len(self.params))]
            for num_of_initials in range(self.n_dim + 1)
        ]

        return np.array(initial_values)

    def _create_initial_value(
        self,
        initial_parameters: Any,
        dim: int,
        num_of_initials: int
    ) -> Any:
        if initial_parameters is not None:
            if isinstance(initial_parameters[dim]['value'], (int, float, np.integer, np.floating)):
                initial_parameters[dim]['value'] = [initial_parameters[dim]['value']]

            if type(initial_parameters[dim]['value']) not in [list, ListConfig]:
                raise TypeError('Default parameter should be set as list.')

            if num_of_initials < len(initial_parameters[dim]['value']):
                val = initial_parameters[dim]['value'][num_of_initials]
                if self.params[dim].type.lower() == 'ordinal':
                    val = np.abs(np.array(self.params[dim].sequence) - val).argmin()
                return val
            else:
                val = self.params[dim].sample(rng=self._rng)['value']
                if self.params[dim].type.lower() == 'ordinal':
                    val = np.abs(np.array(self.params[dim].sequence) - val).argmin()
                return val
        else:
            val = self.params[dim].sample(rng=self._rng)['value']
            if self.params[dim].type.lower() == 'ordinal':
                val = np.abs(np.array(self.params[dim].sequence) - val).argmin()
            return val

    def _add_executing(
        self, y: np.ndarray[Any, Any],
        index: int | None = None
    ) -> None:
        """Add a parameter set to an execution candidate.

        Args:
            y (np.ndarray): A current y values.
            index (int): An index to check Shrink results.

        Returns:
            None
        """
        out_of_boundary = False

        if self._is_out_of_boundary(y):
            out_of_boundary = True
            self.logger.debug(f'_add_executing out of boundary y: {y}')

        vertex_id = generate_random_name(rng=self._rng)
        params = []

        for yi, p in zip(y, self.params):
            params.append({'parameter_name': p.name, 'value': yi})

        self._executing.append(
            {
                'vertex_id': vertex_id,
                'parameters': params,
                'state': self._state,
                'itr': self._evaluated_itr,
                'index': index,
                'out_of_boundary': out_of_boundary
            }
        )

        if out_of_boundary:
            result = float('inf')
            if self._maximize:
                result = float('inf') * -1
            self.add_result_parameters(
                {
                    'vertex_id': vertex_id,
                    'parameters': params,
                    'state': self._state,
                    'itr': self._evaluated_itr,
                    'index': index,
                    'out_of_boundary': out_of_boundary,
                    'result': result
                }
            )

        self._executing_index += 1
        self.search()

    def _add_y_history(self) -> None:
        """Add current y to history.

        Returns:
            None
        """
        if not self._out_of_boundary:
            self._history['evaluated_y'].append(self.y)
        self._history['total_y'].append(self.y)

    def _pop_result(self) -> dict[str, Any] | None:
        """Pop a result.

        Returns:
            dict | None: It returns a result if exists. Otherwise, it
                returns None.

        Raises:
            ValueError: Causes when the name of a popped result is not included
                int _executing list.

            ValueError: Causes when a current state is 'WaitShrink' and out of
                boundary.
        """
        try:
            r = self._result.pop()
        except IndexError:
            return None

        try:
            i = [e['vertex_id'] for e in self._executing].index(r['vertex_id'])
        except ValueError:
            self.logger.error(f'Could not find match for r: {r}')
            raise ValueError(f'Could not find match for r: {r}')

        if r['out_of_boundary']:
            if self._state == 'WaitShrink':
                self.logger.error(f'out of boundary in WaitShrink. r: {r}')
                raise ValueError(f'out of boundary in WaitShrink. r: {r}')
            r['result'] = float('inf')
            self._out_of_boundary = True

        if self._maximize:
            r['result'] *= -1

        self._executing.pop(i)
        return r

    def _change_state(self, state: str) -> None:
        """Change current state.

        Args:
            state (str): A state string.

        Returns:
            None

        Raises:
            ValueError: Causes when unsupported state is given.
        """
        if state not in STATES:
            self.logger.error(f'Unsupported state: {state}')
            raise ValueError(f'Unsupported state: {state}')

        self._state = state

    def _wait_initialize(self, results: list[dict[str, Any]]) -> None:
        """Wait first parameter results are finished.

        Args:
            results (list[dict]): A list of execution results.

        Returns:
            None
        """
        for r in results:
            if self._state == r['state']:
                self.f = np.append(self.f, r['result'])

        if len(self.y) == len(self.f):
            self.f = np.array(self.f)
            self._change_state('Initialize')

    def _initialize(self) -> None:
        """Initialize state method.

        Returns:
            None
        """
        self._centroid()
        self._reflect()

    def _wait_reflect(self, results: list[dict[str, Any]]) -> None:
        """Wait Reflect calculations are finished.

        Args:
            results (list[dict]): A list of execution results.

        Returns:
            None
        """
        for r in results:
            if r['state'] == self._state:
                self._fr = r['result']
                self._change_state('ReflectBranch')

    def _reflect_branch(self) -> None:
        """Branch to change a state after Reflect.

        Returns:
            None
        """
        if self.f[0] <= self._fr < self.f[-2]:
            self.y[-1] = self._storage["r"]
            self.f[-1] = self._fr
            self._finalize()
        elif self._fr < self.f[0]:
            self._expand()
        elif self.f[-2] <= self._fr < self.f[-1]:
            self._outside_contract()
        elif self.f[-1] <= self._fr:
            self._inside_contract()
        else:  # pragma: no cover
            pass  # not reached

    def _wait_expand(self, results: list[dict[str, Any]]) -> None:
        """Wait 'Expand' executions finished.

        Args:
            results (list[dict]): A list of execution results.

        Returns:
            None
        """
        for r in results:
            if r['state'] == self._state:
                self._fe = r['result']
                self._change_state('ExpandBranch')

    def _expand_branch(self) -> None:
        """Branch to change state after 'Expand'.

        Returns:
            None
        """
        if self._fe < self._fr:
            self.y[-1] = self._storage["e"]
            self.f[-1] = self._fe
        else:
            self.y[-1] = self._storage["r"]
            self.f[-1] = self._fr
        self._finalize()

    def _wait_outside_contract(self, results: list[dict[str, Any]]) -> None:
        """Wait the 'OutsideContract' execution finished.

        Args:
            results (list[dict]): A list of execution results.

        Returns:
            None
        """
        for r in results:
            if r['state'] == self._state:
                self._foc = r['result']
                self._change_state('OutsideContractBranch')

    def _outside_contract_branch(self) -> None:
        """Branch to change state after 'OutsideContract'.

        Returns:
            None
        """
        if self._foc <= self._fr:
            self.y[-1] = self._storage["oc"]
            self.f[-1] = self._foc
            self._finalize()
        else:
            self._shrink()

    def _wait_inside_contract(self, results: list[dict[str, Any]]) -> None:
        """Wait the 'InsideContract' execution finished.

        Args:
            results (list[dict]): A list of execution results.

        Returns:
            None
        """
        for r in results:
            if r['state'] == self._state:
                self._fic = r['result']
                self._change_state('InsideContractBranch')

    def _inside_contract_branch(self) -> None:
        """Branch to change state after 'InsideContract'.

        Returns:
            None
        """

        if self._fic < self.f[-1]:
            self.y[-1] = self._storage["ic"]
            self.f[-1] = self._fic
            self._finalize()
        else:
            self._shrink()

    def _wait_shrink(self, results: list[dict[str, Any]]) -> None:
        """Wait the 'Shrink' execution finished.

        Args:
            results (list[dict]): A list of execution results.

        Returns:
            None
        """

        for r in results:
            if r['state'] == self._state:
                self.f[r['index']] = r['result']
                self._num_shrink += 1

        if len(self.y) - 1 == self._num_shrink:
            self._finalize()

    def _finalize(self) -> None:
        """Finalize a current loop.

        Returns:
            None
        """
        if not self._out_of_boundary:
            self._evaluated_itr += 1
            self._history['evaluated_sample'].append(self.y[-1])
        else:
            self.logger.debug(f'history: {self._history["op"]}')
            self.logger.debug(f'y: {self.y}')
            self.logger.debug(f'f: {self.f}')

        self._total_itr += 1
        self._add_y_history()
        self._history['fyr_order'].append(
            np.argsort(np.argsort(self.f))[-1] + 1
        )
        # If do shrink, should save it?
        self._history['total_sample'].append(self.y[-1])

        self._fr = None
        self._fe = None
        self._fic = None
        self._foc = None
        self._num_shrink = 0
        self._out_of_boundary = False
        self._change_state('Initialize')

    def _order_by(self) -> None:
        """Order the values.

        Returns:
            None
        """
        order = np.argsort(self.f)
        self.y = self.y[order]
        self.f = self.f[order]

    def _centroid(self) -> None:
        """Calculate the centroid of points.

        Returns:
            None
        """
        self._storage = {
            "r": None,
            "ic": None,
            "oc": None,
            "e": None,
            "s": None
        }
        self._order_by()
        self.yc = self.y[:-1].mean(axis=0)
        self._history['op'].append('i')

    def _reflect(self) -> None:
        """Compute reflected point.

        Returns:
            None
        """
        yr = self.yc + self.coef["r"] * (self.yc - self.y[-1])
        self._storage["r"] = yr
        self._history['op'].append('r')
        self._change_state('WaitReflect')
        self._add_executing(yr)

    def _expand(self) -> None:
        """Compute the expanded point.

        Returns:
            None
        """
        ye = self.yc + self.coef["e"] * (self.yc - self.y[-1])
        self._storage["e"] = ye
        self._history['op'].append('e')
        self._change_state('WaitExpand')
        self._add_executing(ye)

    def _inside_contract(self) -> None:
        """Compute the contracted point according to ic (inside contract
            coefficient).

        Returns:
            None
        """
        yic = self.yc + self.coef["ic"] * (self.yc - self.y[-1])
        self._storage["ic"] = yic
        self._history['op'].append('ic')
        self._change_state('WaitInsideContract')
        self._add_executing(yic)

    def _outside_contract(self) -> None:
        """Compute the contracted point according to ic (outside contract
            coefficient).

        Returns:
            None
        """
        yoc = self.yc + self.coef["oc"] * (self.yc - self.y[-1])
        self._storage["oc"] = yoc
        self._history['op'].append('io')
        self._change_state('WaitOutsideContract')
        self._add_executing(yoc)

    def _shrink(self) -> None:
        """Shrink all points except y[0].

        Returns:
            None
        """
        for i in range(1, len(self.y)):
            self.y[i] = self.y[0] + self.coef["s"] * (self.y[i] - self.y[0])
            self._change_state('WaitShrink')
            self._add_executing(self.y[i], i)

        self._history['op'].append('s')

    def _is_out_of_boundary(self, y: np.ndarray[Any, Any]) -> bool:
        """Is points out of boundary or not.

        Args:
            y (np.ndarray): Current evaluated points.

        Returns:
            bool: Is points out of boundary or not.
        """
        for yi, b in zip(y, self.bdrys):
            if b[0] > yi or yi > b[1]:
                return True

        return False

    def add_result_parameters(self, result: dict[str, Any]) -> None:
        """Add a new result.

        Args:
            result (dict): A dictionary of a result parameter.

        Returns:
            None
        """
        self._result.append(result)

    def search(self) -> list[dict[str, Any]] | None:
        """Proceed a search step. One search method does not increment the
        iteration. It increments when finalize method is called.

        The details of state transitions are as follows: ::

            -->: state change
            indent: conditional branch

            WaitInitialize: wait first three executing
            --> Initialize: add reflect executing
            --> WaitReflect: wait executing reflect one
            --> ReflectBranch: conditional jump using reflect
                --> back to Initialize
                --> WaitExpand: wait executing expand one
                    --> ExpandBranch: calc using expand
                        ---> back to Initialize
                    --> WaitOutsideContract: wait executing oc one
                        ---> OutsideContractBranch: conditional jump using oc
                            --> back to Initialize
                            --> WaitShrink: wait executing shrink f(y)
                                ---> back to Initialize
                    --> WaitInsideContract: wait executing ic one
                        ---> InsideContractBranch: conditional jump using ic
                            --> back to Initialize
                            --> WaitShrink: wait executing shrink f(y)
                                ---> back to Initialize

        Returns:
            None
        """
        if self._evaluated_itr >= self._max_itr:
            return None

        results = []

        while True:
            r = self._pop_result()
            if r is None:
                break
            results.append(r)

        if self._state == 'WaitInitialize':
            self._wait_initialize(results)
        elif self._state == 'Initialize':
            self._initialize()
        elif self._state == 'WaitReflect':
            self._wait_reflect(results)
        elif self._state == 'ReflectBranch':
            self._reflect_branch()
        elif self._state == 'WaitExpand':
            self._wait_expand(results)
        elif self._state == 'ExpandBranch':
            self._expand_branch()
        elif self._state == 'WaitOutsideContract':
            self._wait_outside_contract(results)
        elif self._state == 'OutsideContractBranch':
            self._outside_contract_branch()
        elif self._state == 'WaitInsideContract':
            self._wait_inside_contract(results)
        elif self._state == 'InsideContractBranch':
            self._inside_contract_branch()
        elif self._state == 'WaitShrink':
            self._wait_shrink(results)
        else:
            self.logger.error(f'Invalid state: {self._state}')
            raise ValueError(f'Invalid state: {self._state}')

        return self._executing
