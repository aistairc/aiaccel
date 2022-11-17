from aiaccel.evaluator.maximize_evaluator import MaximizeEvaluator

from tests.base_test import BaseTest


class TestMaximizeEvaluator(BaseTest):
    def test_maximize_evaluator(self):
        options = {
            'config': str(self.config_json),
            'resume': None,
            'clean': False,
            'fs': False,
        }
        evaluator = MaximizeEvaluator(options['config'])
        evaluator.evaluate()
        assert evaluator.hp_result is None
