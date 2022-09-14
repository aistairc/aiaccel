from aiaccel.master.evaluator.abstract import AbstractEvaluator
from tests.base_test import BaseTest
import aiaccel


class TestAbstractEvaluator(BaseTest):

    def test_init(self):
        options = {
            'config': str(self.config_json),
            'resume': None,
            'clean': False,
            'process_name': 'test'
        }
        evaluator = AbstractEvaluator(options)
        assert evaluator.hp_result is None

    def test_evaluate(self):
        options = {
            'config': str(self.config_json),
            'resume': None,
            'clean': False,
            'process_name': 'test'
        }
        evaluator = AbstractEvaluator(options)
        evaluator.storage.alive.init_alive()
        assert evaluator.evaluate() is None

        # try:
        #     evaluator.evaluate()
        #     assert False
        # except NotImplementedError:
        #     assert True

    def test_print(self, clean_work_dir, setup_hp_finished, work_dir):
        options = {
            'config': str(self.config_json),
            'resume': None,
            'clean': False,
            'process_name': 'test'
        }
        
        evaluator = AbstractEvaluator(options)
        setup_hp_finished(1)
        evaluator.hp_result = work_dir.joinpath(
            aiaccel.dict_hp_finished, f'001.{aiaccel.extension_hp}')
        assert evaluator.print() is None

    def test_save(self, clean_work_dir, setup_hp_finished, work_dir):
        options = {
            'config': str(self.config_json),
            'resume': None,
            'clean': False,
            'process_name': 'test'
        }
        
        evaluator = AbstractEvaluator(options)
        setup_hp_finished(1)
        evaluator.hp_result = work_dir.joinpath(
            aiaccel.dict_hp_finished, f'001.{aiaccel.extension_hp}')
        evaluator.save()
        assert work_dir.joinpath(aiaccel.dict_result, aiaccel.file_final_result).exists()
