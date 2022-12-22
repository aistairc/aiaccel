from aiaccel.master.evaluator.maximize_evaluator import MaximizeEvaluator
from aiaccel.config import load_config

from tests.base_test import BaseTest


class TestMaximizeEvaluator(BaseTest):
    def test_maximize_evaluator(self):
        evaluator = MaximizeEvaluator(self.configs["config.json"])
        evaluator.evaluate()
        assert evaluator.hp_result is None
