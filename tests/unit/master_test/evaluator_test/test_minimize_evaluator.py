from aiaccel.master.evaluator.minimize_evaluator import MinimizeEvaluator

from tests.base_test import BaseTest


class TestMinimizeEvaluator(BaseTest):
    def test_maximize_evaluator(self):
        options = {
            'config': self.config_json,
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'master'
        }
        evaluator = MinimizeEvaluator(options)
        evaluator.evaluate()
        assert evaluator.hp_result is None
