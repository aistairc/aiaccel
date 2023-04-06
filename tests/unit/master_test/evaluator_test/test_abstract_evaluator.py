from aiaccel.common import dict_hp_finished
from aiaccel.common import dict_result
from aiaccel.common import extension_hp
from aiaccel.common import file_final_result
from aiaccel.master import AbstractEvaluator
from tests.base_test import BaseTest


class TestAbstractEvaluator(BaseTest):

    def test_init(self):
        evaluator = AbstractEvaluator(self.load_config_for_test(self.configs["config.json"]))
        assert evaluator.hp_result is None

    def test_evaluate(self):
        evaluator = AbstractEvaluator(self.load_config_for_test(self.configs["config.json"]))
        assert evaluator.evaluate() is None

        # try:
        #     evaluator.evaluate()
        #     assert False
        # except NotImplementedError:
        #     assert True

    def test_print(self, clean_work_dir, setup_hp_finished, work_dir):
        evaluator = AbstractEvaluator(self.load_config_for_test(self.configs["config.json"]))
        setup_hp_finished(1)
        evaluator.hp_result = work_dir.joinpath(
            dict_hp_finished, f'001.{extension_hp}')
        assert evaluator.print() is None

    def test_save(self, clean_work_dir, setup_hp_finished, work_dir):
        evaluator = AbstractEvaluator(self.load_config_for_test(self.configs["config.json"]))
        setup_hp_finished(1)
        evaluator.hp_result = {}
        evaluator.save()
        assert work_dir.joinpath(dict_result, file_final_result).exists()
