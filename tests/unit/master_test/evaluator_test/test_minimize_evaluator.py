from aiaccel.master.evaluator.minimize_evaluator import MinimizeEvaluator
from tests.base_test import BaseTest

# # def test_maximize_evaluator(load_test_config):
# # コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(荒本)
# def test_maximize_evaluator(load_test_config):
#     # config = load_test_config()
#     # コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(荒本)
#     config = load_test_config()
#     # evaluator = MinimizeEvaluator(config)
#     # コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(荒本)
#     evaluator = MinimizeEvaluator(config)
#     evaluator.evaluate()
#     assert evaluator.hp_result is None

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
