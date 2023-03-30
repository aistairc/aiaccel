from aiaccel.master import AbstractEvaluator
from tests.base_test import BaseTest


class TestAbstractEvaluator(BaseTest):

    def test_init(self):
        options = {
            'config': str(self.config_json),
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'test'
        }
        evaluator = AbstractEvaluator(options)
        assert evaluator.hp_result is None

    def test_evaluate(self):
        options = {
            'config': str(self.config_json),
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'test'
        }
        evaluator = AbstractEvaluator(options)
        assert evaluator.evaluate() is None

        # try:
        #     evaluator.evaluate()
        #     assert False
        # except NotImplementedError:
        #     assert True

    def test_print(self, setup_hp_finished, work_dir):
        options = {
            'config': str(self.config_json),
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'test'
        }

        evaluator = AbstractEvaluator(options)
        setup_hp_finished(1)
        evaluator.evaluate()
        assert evaluator.print() is None

    def test_save(self, setup_hp_finished, work_dir):
        options = {
            'config': str(self.config_json),
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'test'
        }

        evaluator = AbstractEvaluator(options)
        setup_hp_finished(1)
        evaluator.evaluate()
        assert evaluator.save() is None
