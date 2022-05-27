from aiaccel.master.evaluator.abstract_evaluator import AbstractEvaluator
from aiaccel.parameter import get_best_parameter
from aiaccel.util.filesystem import get_file_hp_finished
import aiaccel


class MaximizeEvaluator(AbstractEvaluator):
    """A evaluator class to maximize the results.

    """

    def evaluate(self) -> None:
        """Run an evaluation.

        Returns:
            None
        """
        files = get_file_hp_finished(self.ws)
        best, best_file = get_best_parameter(
            files,
            aiaccel.goal_maximize,
            self.dict_lock
        )

        self.hp_result = best_file
