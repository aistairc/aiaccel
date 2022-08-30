from aiaccel.util.name import generate_random_name
from aiaccel.parameter import HyperParameter
from typing import Dict, List, Optional, Union
import copy
import logging
import numpy as np
import random

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

hptype = Union[
    List[Dict[str, Union[str, Union[float, List[float]]]]],
    None
]


class NelderMead(object):
    """A class implementing Nelder-Mead method.

    Attributes:
        bdrys (np.ndarray): A list of boundaries.
        coef (Dict[str, float]): A dictionary of coefficients.
        f (np.ndarray): A list of evaluated parameter results.
        logger (logging.Logger): A logger object.
        params (List[HyperParameter]): A list of hyper parameters.
        storage (Dict[str, Union[float, None]]): A dictionary to store temporal
            calculation results.
        y (np.ndarray): A list of current evaluated parameters.
        yc (float): A current centroid value of y.
        _evaluated_itr (int): A count of evaluation each loop.
        _executing (List[dict]): A list to store candidates to be executed.
        _executing_index (int): A number to be added to executing list.
        _fe (float): A temporal result of Expand.
        _fic (float): A temporal result of Inside Contraction.
        _foc (float): A temporal result of Outside Contraction.
        _fr (float): A temporal result of Reflection
        _history (Dict[str, List[Union[float, str]]]): A storage of execution
            history of each value and operator.
        _maximize (bool): Evaluate the result as maximize or minimize.
        _max_itr (int): A number of max iterations. This is compared with
            _evaluated_itr.
        _num_shrink (int): A number of Shrink.
        _out_of_boundary (bool): Is a current iteration out of boundaries or
            not.
        _result (List[float]): A list of results for _executing.
        _state (str): A current state.
        _total_itr (int): A number of iterations. Currently same with
            _evaluated_itr. It's different if counts out of boundaries.

    ToDo: Fix float comparison errors.
    """

    def __init__(
        self,
        params: List[HyperParameter],
        iteration: Optional[float] = float('inf'),
        coef: Optional[Union[dict, None]] = None,
        maximize: Optional[bool] = False,
        initial_parameters: Optional[hptype] = None
    ) -> None:
        """Initial method of NelderMead.

        Args:
            -params: A list of hyper parameter objects.
            -iteration: A max iteration counts.
            -coef: A coefficient values.
            -maximize: Evaluate maximize or not.
            -initial_parameters: A initial parameters.
        """
        if coef is None:
            coef = {"r": 1.0, "ic": - 0.5, "oc": 0.5, "e": 2.0, "s": 0.5}

        self.logger = logging.getLogger('root.optimizer.nelder_mead')
        self.params = params
        self.bdrys = np.array([[p.lower, p.upper] for p in self.params])
        self.coef = coef
        self.n_dim = len(self.bdrys)

        self.y = self._create_initial_values()
        if initial_parameters is not None:
            for dp in initial_parameters:
                if (
                    type(dp['value']) is int or
                    type(dp['value']) is float
                ):
                    dp['value'] = [dp['value']]

                if type(dp['value']) is not list:
                    raise TypeError('Default parameter should be set as list.')

                ind = [p.name for p in self.params].index(dp['parameter_name'])

                for i in range(len(dp['value'])):
                    self.y[i][ind] = dp['value'][i]

        self.f = []
        self.yc = None
        self._storage = {
            "r": None, "ic": None, "oc": None, "e": None, "s": None}
        self._max_itr = iteration
        self._total_itr = 0         # included out of boundary
        self._evaluated_itr = 0     # not included out of boundary
        self._history = {
            'total_y': [],              # y each loop
            'evaluated_y': [],          # y each loop not included out of boundary
            'op': [],                   # operations such as 'reflect' and so on.
            'total_sample': [],         # sampled point each loop
            'evaluated_sample': [],     # sampled point each loop not included out of boundary
            'fyr_order': []             # order of f(yr) in self.f
        }
        self._executing_index = 0
        self._executing = []
        self._fr = None
        self._fe = None
        self._fic = None
        self._foc = None
        self._maximize = maximize
        self._num_shrink = 0
        self._state = 'WaitInitialize'
        self._out_of_boundary = False
        self._result = []
        for y in self.y:
            self._add_executing(y)

    def _create_initial_values(self) -> list:
        initial_values = [
            [random.random() * (b[1] - b[0]) + b[0] for b in self.bdrys]
            for _ in range(self.n_dim + 1)
        ]
        return np.array(initial_values)

    def _add_executing(
        self, y: np.ndarray,
        index: Optional[int] = None
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
            self.logger.debug('_add_executing out of boundary y: {}'.format(y))

        vertex_id = generate_random_name()
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

    def _pop_result(self) -> Union[dict, None]:
        """Pop a result.

        Returns:
            Union[dict, None]: It returns a result if exists. Otherwise, it
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
            self.logger.error('Unsupported state: {}'.format(state))
            raise ValueError('Unsupported state: {}'.format(state))

        self._state = state

    def _wait_initialize(self, results: List[dict]) -> None:
        """Wait first parameter results are finished.

        Args:
            results (List[dict]): A list of execution results.

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

    def _wait_reflect(self, results: List[dict]) -> None:
        """Wait Reflect calculations are finished.

        Args:
            results (List[dict]): A list of execution results.

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

    def _wait_expand(self, results: List[dict]) -> None:
        """Wait 'Expand' executions finished.

        Args:
            results (List[dict]): A list of execution results.

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

    def _wait_outside_contract(self, results: List[dict]) -> None:
        """Wait the 'OutsideContract' execution finished.

        Args:
            results (List[dict]): A list of execution results.

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

    def _wait_inside_contract(self, results: List[dict]) -> None:
        """Wait the 'InsideContract' execution finished.

        Args:
            results (List[dict]): A list of execution results.

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

    def _wait_shrink(self, results) -> None:
        """Wait the 'Shrink' execution finished.

        Args:
            results (List[dict]): A list of execution results.

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
            self.logger.debug('history: {}'.format(self._history['op']))
            self.logger.debug('y: {}'.format(self.y))
            self.logger.debug('f: {}'.format(self.f))

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

    def _is_out_of_boundary(self, y: np.ndarray) -> bool:
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

    def add_result_parameters(self, result: dict) -> None:
        """Add a new result.

        Args:
            result (dict): A dictionary of a result parameter.

        Returns:
            None
        """
        self._result.append(result)

    def search(self) -> Union[List[dict], None]:
        """Proceed a search step. One search method does not increment the
            iteration. It increments when finalize method is called.

        The details of state transitions are as follows:

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
            self.logger.error('Invalid state: {}'.format(self._state))
            raise ValueError('Invalid state: {}'.format(self._state))

        return self._executing

    def serialize(self) -> dict:
        """Serialize this module.

        Returns:
            dict: The serialized master objects.
        """
        storage = {}
        for key in ['e', 'ic', 'oc', 'r', 's']:
            if type(self._storage[key]) is np.ndarray:
                storage[key] = self._storage[key].tolist()
            elif type(self._storage[key]) is np.float64:
                storage[key] = float(self._storage[key])
            else:
                storage[key] = self._storage[key]

        y = self.y
        f = self.f
        yc = self.yc

        if type(self.y) is np.ndarray:
            y = self.y.tolist()

        if type(self.f) is np.ndarray:
            f = self.f.tolist()

        if type(self.yc) is np.ndarray:
            yc = self.yc.tolist()

        history = {'op': self._history['op']}
        for key in [
            'total_y',
            'evaluated_y',
            'total_sample',
            'evaluated_sample',
            'fyr_order'
        ]:
            value = []

            for v in self._history[key]:
                if type(v) is np.ndarray:
                    value.append(v.tolist())
                elif type(v) is np.float64:
                    value.append(float(v))
                elif type(v) is np.int64:
                    value.append(int(v))
                else:
                    print('nm serialize history key: ', key, ', type: ', type(v))
                    value.append(v)
            history[key] = value

        executing = copy.deepcopy(self._executing)

        for e in executing:
            for ep in e['parameters']:
                if type(ep['value']) is np.float64:
                    ep['value'] = float(ep['value'])

        return {
            'coef': self.coef,
            'y': y,
            'f': f,
            'yc': yc,
            '_executing': executing,
            'storage': storage,
            '_total_itr': self._total_itr,
            '_evaluated_itr': self._evaluated_itr,
            '_history': history,
            '_executing_index': self._executing_index,
            '_fr': self._fr,
            '_fe': self._fe,
            '_fic': self._fic,
            '_foc': self._foc,
            '_num_shrink': self._num_shrink,
            '_state': self._state,
            '_out_of_boundary': self._out_of_boundary,
            '_result': self._result
        }

    def deserialize(self, dict_objects: dict) -> None:
        """Deserialize this module.

        Args:
            dict_objects(dict): A dictionary including serialized objects.

        Returns:
            None
        """
        storage = {}
        for key in ['e', 'ic', 'oc', 'r', 's']:
            if type(dict_objects['storage'][key]) is list:
                storage[key] = np.array(dict_objects['storage'][key])
            else:
                storage[key] = dict_objects['storage'][key]

        y = dict_objects['y']
        if type(y) is list:
            y = np.array(y)

        f = dict_objects['f']
        if type(f) is list:
            f = np.array(f)

        yc = dict_objects['yc']
        if type(yc) is list:
            yc = np.array(yc)

        history = {
            'total_y': [],
            'evaluated_y': [],
            'op': dict_objects['_history']['op'],
            'total_sample': [],
            'evaluated_sample': [],
            'fyr_order': []
        }

        for key in [
            'total_y',
            'evaluated_y',
            'total_sample',
            'evaluated_sample',
            'fyr_order'
        ]:
            for v in dict_objects['_history'][key]:
                if type(v) is list:
                    history[key].append(
                        np.array(v)
                    )
                elif type(v) is float:
                    history[key].append(
                        np.float64(v)
                    )
                elif type(v) is int:
                    history[key].append(
                        np.int64(v)
                    )
                else:
                    history[key].append(v)

        executing = copy.deepcopy(dict_objects['_executing'])
        for e in executing:
            for ep in e['parameters']:
                if type(ep['value']) is float:
                    ep['value'] = np.float64(ep['value'])

        self.y = y
        self.f = f
        self.yc = yc
        self._storage = storage
        self._executing = executing
        self._history = history
        self.coef = dict_objects['coef']
        self._total_itr = dict_objects['_total_itr']
        self._evaluated_itr = dict_objects['_evaluated_itr']
        self._executing_index = dict_objects['_executing_index']
        self._fr = dict_objects['_fr']
        self._fe = dict_objects['_fe']
        self._fic = dict_objects['_fic']
        self._foc = dict_objects['_foc']
        self._num_shrink = dict_objects['_num_shrink']
        self._state = dict_objects['_state']
        self._out_of_boundary = dict_objects['_out_of_boundary']
        self._result = dict_objects['_result']
