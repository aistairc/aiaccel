from aiaccel.master.evaluator.minimize_evaluator import MinimizeEvaluator


# def test_maximize_evaluator(load_test_config):
# コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(荒本)
def test_maximize_evaluator(load_test_config):
    # config = load_test_config()
    # コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(荒本)
    config = load_test_config()
    # evaluator = MinimizeEvaluator(config)
    # コンフィグファイルの読取り形式変更改修に伴いテストコードも変更(荒本)
    evaluator = MinimizeEvaluator(config)
    evaluator.evaluate()
    assert evaluator.hp_result is None
