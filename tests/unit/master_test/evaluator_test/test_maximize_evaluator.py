from aiaccel.command_line_options import CommandLineOptions
from aiaccel.master import MaximizeEvaluator

from tests.base_test import BaseTest


class TestMaximizeEvaluator(BaseTest):
    def test_maximize_evaluator(self):
        options = CommandLineOptions(
            config=str(self.config_json),
            resume=None,
            clean=False,
        )
        evaluator = MaximizeEvaluator(options)
        evaluator.evaluate()
        assert evaluator.hp_result is None
