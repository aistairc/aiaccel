from aiaccel.master.evaluator.maximize_evaluator import MaximizeEvaluator


# def test_maximize_evaluator(load_test_config):
# コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(荒本)
def test_maximize_evaluator(load_test_config):
    # config = load_test_config()
    # コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(荒本)
    config = load_test_config()
    # evaluator = MaximizeEvaluator(config)
    # コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(荒本)
    evaluator = MaximizeEvaluator(config)
    evaluator.evaluate()
    assert evaluator.hp_result is None
