import aiaccel
from aiaccel.master.evaluator.abstract_evaluator import AbstractEvaluator
from aiaccel.config import load_config

from tests.base_test import BaseTest


class TestAbstractEvaluator(BaseTest):

    def test_init(self):
        evaluator = AbstractEvaluator(self.configs["config.json"])
        assert evaluator.hp_result is None

    def test_evaluate(self):
        evaluator = AbstractEvaluator(self.configs["config.json"])
        assert evaluator.evaluate() is None

        # try:
        #     evaluator.evaluate()
        #     assert False
        # except NotImplementedError:
        #     assert True

<<<<<<< HEAD
    def test_print(self, clean_work_dir, setup_hp_finished, work_dir):
        evaluator = AbstractEvaluator(self.configs["config.json"])
=======
    def test_print(self, setup_hp_finished, work_dir):
        options = {
            'config': str(self.config_json),
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'test'
        }

        evaluator = AbstractEvaluator(options)
>>>>>>> 392d1634b3b761e737cfcbca38507b668d7ab129
        setup_hp_finished(1)
        evaluator.hp_result = work_dir.joinpath(
            aiaccel.dict_hp_finished, f'001.{aiaccel.extension_hp}')
        assert evaluator.print() is None

<<<<<<< HEAD
    def test_save(self, clean_work_dir, setup_hp_finished, work_dir):
        evaluator = AbstractEvaluator(self.configs["config.json"])
=======
    def test_save(self, setup_hp_finished, work_dir):
        options = {
            'config': str(self.config_json),
            'resume': None,
            'clean': False,
            'fs': False,
            'process_name': 'test'
        }

        evaluator = AbstractEvaluator(options)
>>>>>>> 392d1634b3b761e737cfcbca38507b668d7ab129
        setup_hp_finished(1)
        evaluator.hp_result = {}
        evaluator.save()
        assert work_dir.joinpath(aiaccel.dict_result, aiaccel.file_final_result).exists()
