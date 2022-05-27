from aiaccel.master.evaluator.abstract_evaluator import AbstractEvaluator
from aiaccel.parameter import get_best_parameter
from aiaccel.util.filesystem import get_file_hp_finished
import aiaccel


class MinimizeEvaluator(AbstractEvaluator):
    """A evaluator class to minimize the results.

    """

    def evaluate(self):
        """Run an evaluation.

        Returns:
            None
        """
        files = get_file_hp_finished(self.ws)
        best, best_file = get_best_parameter(
            files,
            aiaccel.goal_minimize,
            self.dict_lock
        )

        self.hp_result = best_file
