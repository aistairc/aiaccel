from aiaccel.master.evaluator.abstract_evaluator import AbstractEvaluator
from tests.base_test import BaseTest
import aiaccel


class TestAbstractEvaluator(BaseTest):

    def test_init(self):
        evaluator = AbstractEvaluator(self.config)
        assert evaluator.hp_result is None

    def test_evaluate(self):
        evaluator = AbstractEvaluator(self.config)

        try:
            evaluator.evaluate()
            assert False
        except NotImplementedError:
            assert True

    def test_print(self, clean_work_dir, setup_hp_finished, work_dir):
        evaluator = AbstractEvaluator(self.config)
        setup_hp_finished(1)
        evaluator.hp_result = work_dir.joinpath(
            aiaccel.dict_hp_finished, '001.{}'.format(aiaccel.extension_hp))
        assert evaluator.print() is None

    def test_save(self, clean_work_dir, setup_hp_finished, work_dir):
        evaluator = AbstractEvaluator(self.config)
        setup_hp_finished(1)
        evaluator.hp_result = work_dir.joinpath(
            aiaccel.dict_hp_finished,'001.{}'.format(aiaccel.extension_hp))
        evaluator.save()
        assert work_dir.joinpath(aiaccel.dict_result, aiaccel.file_final_result).exists()
