from aiaccel.master import MaximizeEvaluator
from tests.base_test import BaseTest


class TestMaximizeEvaluator(BaseTest):
    def test_maximize_evaluator(self):
        evaluator = MaximizeEvaluator(self.load_config_for_test(self.configs["config.json"]))
        evaluator.evaluate()
        assert evaluator.hp_result[0] is None
