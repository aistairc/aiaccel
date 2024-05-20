# NelderMeadSampler の examples

## 1. ファイル構成

### example.py

- 一般的な NelderMeadSampler の使い方を示したコードです.

### example_parallel.py

- 並列実行時の NelderMeadSampler の使い方を示したコードです.
- NelderMeadSampler の引数 block=True, study.optimize の引数 n_jpbs=3 として、並列実行を有効にしています.

### example_enqueue.py

- optuna.study.enqueue_trial 利用時の NelderMeadSampler の使い方を示したコードです.
- ask-tell インタフェースを利用し、 NelderMeadSampler がパラメータの出力に失敗した時に、ランダムなパラメータを enqueue_trial で探索しています.

### example_sub_sampler.py

- sub_sampler 機能の利用時の NelderMeadSampler の使い方を示したコードです.
- NelderMeadSampler の引数 sub_sampler=optuna.samplers.TPESampler として、NelderMeadSampler がパラメータの出力に失敗した時に、TPESampler で探索しています.
- sub_sampler 機能の利用時は、並列であっても引数 block=False にする必要があります.

### compare_optimizer_ackley.py

- TPE, NelderMead, NelderMead(sub_sampler=optuna.samplers.TPESampler) の3種類の最適化手法を Ackley 関数を用いて比較、及び比較結果の可視化を行うコードです.

## 2. 動作説明

- aiaccel のインストール・仮想環境の activate 後に、該当ファイルを実行してください.

```bash
python example.py
```

- compare_optimizer_ackley.py の実行には、 matplotlib のインストールが必要です.

```bash
pip install matplotlib pandas
```

## 3. 結果の確認

- example コードの実行結果は、標準出力に表示されます.
- compare_optimizer_ackley.py の実行結果は、 ackley_100step_10parallel ackley_1000step_series ディレクトリに保存されます.