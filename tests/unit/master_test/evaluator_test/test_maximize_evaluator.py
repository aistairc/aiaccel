from aiaccel.master.evaluator.maximize import MaximizeEvaluator
from tests.base_test import BaseTest

# # def test_maximize_evaluator(load_test_config):
# # コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(荒本)
# def test_maximize_evaluator(load_test_config):
#     # config = load_test_config()
#     # コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(荒本)
#     config = load_test_config()
#     # evaluator = MaximizeEvaluator(config)
#     # コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(荒本)
#     evaluator = MaximizeEvaluator(config)
#     evaluator.evaluate()
#     assert evaluator.hp_result is None


class TestMaximizeEvaluator(BaseTest):
    def test_maximize_evaluator(self):
        options = {
            'config': str(self.config_json),
            'resume': None,
            'clean': False,
        }
        evaluator = MaximizeEvaluator(options)
        evaluator.evaluate()
        assert evaluator.hp_result is None
