from aiaccel.command_line_options import CommandLineOptions
from aiaccel.master import MinimizeEvaluator

from tests.base_test import BaseTest


class TestMinimizeEvaluator(BaseTest):
    def test_maximize_evaluator(self):
        options = CommandLineOptions(
            config=str(self.config_json),
            resume=None,
            clean=False,
            process_name="master"
        )
        evaluator = MinimizeEvaluator(options)
        evaluator.evaluate()
        assert evaluator.hp_result is None
