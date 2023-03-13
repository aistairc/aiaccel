from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Literal

import aiaccel
from aiaccel.config import Config
from aiaccel.storage.storage import Storage
from aiaccel.util.filesystem import create_yaml


class AbstractVerification(object):
    """An abstract class of verification.

    Args:
        options (dict[str, str | int | bool]): A dictionary containing
            command line options as well as process name.

    Attributes:
        options (dict[str, str | int | bool]): A dictionary containing
            command line options as well as process name.
        config (Config): Config object
        ws (Path): Path to the workspace.
        dict_lock (Path): Path to "lock", i.e. `ws`/lock.
        is_verified (bool): Whether verified or not.
        condition (list[dict[str, int | float]]): A list of verification
            conditions. Each element dict has keys 'loop', 'minimum', and
            'maximum' with values as int, float, and float, respectively.
            The verification is `True` if the optimized result of which the
            trial id is the given 'loop' is greater than or equal to
            'minimum' as well as is less than or equal to the 'maximum'.
        verification_result (list[dict]): Deepcopy object of condition.
        storage (Storage): Storage object.
    """

    def __init__(self, options: dict[str, str | int | bool]) -> None:
        # === Load config file===
        self.options = options
        self.config = Config(self.options['config'])
        self.ws = Path(self.config.workspace.get()).resolve()
        self.dict_lock = self.ws / aiaccel.dict_lock
        self.is_verified = self.config.is_verified.get()
        self.condition = self.config.condition.get()
        self.verification_result = copy.copy(self.condition)
        self.storage = Storage(self.ws)

        if self.config.goal.get().lower() == 'minimize':
            self._current_best_start = float('inf')
            self._comparator = min
        else:
            self._current_best_start = float('-inf')
            self._comparator = max
        self._verified_loops = []
        self._verified_trial_ids = []

    def verify(self) -> None:
        """Run a verification.

        The trigger to run a verification, is described in configuration file
        'verification' > 'conditions'.
        """
        if not self.is_verified:
            return

        # TODO: Flatten following for-loop if main process is flatten.
        for condition_id, target_condition in enumerate(self.condition):
            loop = target_condition['loop']
            if not self._is_loop_verifiable(loop):
                continue
            if self._is_loop_verified(loop):
                continue
            finished_trial_ids = self.storage.get_finished()
            finished_trial_ids.sort()
            current_best = self._find_best_objective_before_target_loop(
                finished_trial_ids,
                loop
            )
            if self._make_verification(current_best, condition_id) == 'verified':
                self._verified_loops.append(loop)
                self.save(loop)

    def _is_loop_verifiable(self, loop: int) -> bool:
        return loop < self.config.trial_number.get()

    def _is_loop_verified(self, loop: int) -> bool:
        return loop in self._verified_loops

    def _find_best_objective_before_target_loop(
        self,
        finished_trial_ids: list[int],
        loop: int
    ) -> float:
        current_best = self._current_best_start
        self._verified_trial_ids = []
        for trial_id in finished_trial_ids:
            if trial_id > loop:
                break
            result = self.storage.result.get_any_trial_objective(trial_id)
            current_best = self._comparator(current_best, result)
            self._verified_trial_ids.append(trial_id)
        return current_best

    def _make_verification(self, current_best: float, condition_id: int) -> Literal['verified', '']:
        """Run a verification.

        Args:
            current_best (float): Best objective before target loop.
            condition_id (int): Index of target condition.

        Returns:
            str: String which indicates whether verification was made.
                'verified' if verification was made, and '' if it was not.
        """
        loop = self.condition[condition_id]['loop']
        lower = self.condition[condition_id]['minimum']
        upper = self.condition[condition_id]['maximum']
        if lower <= current_best <= upper:
            self.verification_result[condition_id]['passed'] = True
            return 'verified'
        elif len(self._verified_trial_ids) == loop + 1:
            self.verification_result[condition_id]['passed'] = False
            return 'verified'
        else:
            return ''

    def load_verification_config(self) -> None:
        """Load configurations about verification.

        Returns:
            None
        """
        self.is_verified = self.config.is_verified.get()
        self.condition = self.config.condition.get()
        self.verification_result = copy.copy(self.condition)

    def print(self) -> None:
        """Print current verifications result.

        Returns:
            None
        """
        if not self.is_verified:
            return None

        logger = logging.getLogger('root.master.verification')
        logger.info('Current verification is followings:')
        logger.info(f'{self.verification_result}')

    def save(self, name: int) -> None:
        """Save current verifications result to a file.

        Args:
            name (int):

        Returns:
            None
        """
        if not self.is_verified:
            return None

        path = self.ws / aiaccel.dict_verification / f'{name}.{aiaccel.extension_verification}'
        create_yaml(path, self.verification_result, self.dict_lock)
        logger = logging.getLogger('root.master.verification')
        logger.info(f'Save verifiation file: {name}.{aiaccel.extension_verification}')
