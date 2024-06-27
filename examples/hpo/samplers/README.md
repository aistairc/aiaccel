# NelderMeadSampler の examples

## 1. ファイル構成

### example.py

- 一般的な NelderMeadSampler の使い方を示したコードです.
- 最適化対象はベンチマーク関数 shpere になっています.(以下の example も記述が無い場合は同様)

### example_parallel.py

- 並列実行時の NelderMeadSampler の使い方を示したコードです.
- NelderMeadSampler の引数 block=True, study.optimize の引数 n_jobs=3 として、並列実行を有効にしています.
- 並列実行を有効にすることで、初期点計算と shrink 時の計算を並列化でき、直列実行と比べて高速化できます.

### example_enqueue.py

- optuna.study.enqueue_trial 利用時の NelderMeadSampler の使い方を示したコードです.
- ask-tell インタフェースを利用し、 NelderMeadSampler がパラメータの出力に失敗した時に、ランダムなパラメータを enqueue_trial で探索しています.

### example_sub_sampler.py

- sub_sampler 機能の利用時の NelderMeadSampler の使い方を示したコードです.
- NelderMeadSampler の引数 sub_sampler=optuna.samplers.TPESampler として、NelderMeadSampler がパラメータの出力に失敗した時に、TPESampler で探索しています.
- sub_sampler 機能の利用時は、並列であっても引数 block=False にする必要があります. (block=False でも並列実行は可能です.)

### coco

- ブラックボックス最適化評価用フレームワーク coco を用いた NelderMeadSampler の検証用コードを含んだディレクトリです.
- 詳細は該当ディレクトリ内の README.md を参照してください.

## 2. 動作説明

- aiaccel のインストール・仮想環境の activate 後に、該当ファイルを実行してください.

```bash
python example.py
```

## 3. 結果の確認

- example コードの実行結果は、標準出力に表示されます.
