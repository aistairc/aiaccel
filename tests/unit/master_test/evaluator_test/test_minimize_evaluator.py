from aiaccel.master import MinimizeEvaluator
from tests.base_test import BaseTest


class TestMinimizeEvaluator(BaseTest):
    def test_maximize_evaluator(self):
        evaluator = MinimizeEvaluator(self.load_config_for_test(self.configs["config.json"]))
        evaluator.evaluate()
        assert evaluator.hp_result[0] is None
