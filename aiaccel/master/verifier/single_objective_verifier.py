from __future__ import annotations

from typing import Literal
from typing import Any

from aiaccel.master.verifier import AbstractVerifier


class SingleObjectiveVerifier(AbstractVerifier):
    """Verifier for single-objective optimization.
    """

    def __init__(self, options: dict[str, Any]) -> None:
        super().__init__(options)
        if self.config.goal.get().lower() == 'minimize':
            self._current_best_start = float('inf')
            self._comparator = min
        else:
            self._current_best_start = float('-inf')
            self._comparator = max
        self._verified_loops: list[int] = []
        self._verified_trial_ids: list[int] = []

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
