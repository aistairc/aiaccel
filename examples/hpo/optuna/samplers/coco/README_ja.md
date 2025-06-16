# coco を利用した NelderMeadSampler の検証用コード

## 1. ファイル構成

### nelder-mead
### nelder-mead-subTPE
### TPE

- 各 sampler の最適化結果の csv が格納されるディレクトリです.

### experiment_coco.py

- coco を用いて検証を行う本体のコードです.
- 次元数*20　step、10並列での実行を想定しています.
- 実行すると optuna の結果した結果が `optuna_csv` に、並列ステップ毎の結果が `step_csv` に出力されます.

### main_parallel_coco.py

- `job_dispatcher` を用いて各 sampler ・関数・次元毎にジョブを投入するコードです.

### objective.sh

- `job_dispatcher` で投入する qsub 用のスクリプトです.

### plot.py

- matplotlib を用いて各 sampler の結果をグラフ化するコードです.
- 各 sampler のディレクトリの `optuna_csv` を参照します.

### result_bbob_dim_vs_value-fopt_parallel.png

- `plot.py` を実行して出力した検証結果を可視化したグラフ画像です.
- 横軸 次元数 縦軸 最適化結果の平均・偏差 のグラフが、ベンチマーク関数24個分並んでいます.

## 2. 動作説明

- aiaccel のインストール・仮想環境の activate を行ってください.

- coco インストールを行ってください.
  - 詳細は下記 git を参照してください.
    https://github.com/numbbo/coco

- `objective.sh` の仮想環境、 `main_parallel_coco.py` の job_group を適切なパス・ ID に書き換えてください.
- `main_parallel.py` を実行すると、各 sampler の検証が実行されます.
- 結果は各ディレクトリ直下の `optuna_csv`, `step_csv` に保存されます.

```bash
cd nelder-mead
python main_parallel.py
```

- `plot.py` の実行には、pandas, matplotlib のインストールが必要です.

```bash
pip install pandas matplotlib
python plot.py
```

## 3. 結果の確認

- 各 sampler の検証結果は sampler に対応したディレクトリ以下の `optuna_csv`, `step_csv` に出力されます.
- `plot.py` の可視化結果は `result_bbob_dim_vs_value-fopt_parallel.png` に出力されます.
  - 可視化結果からは、並列実行時には nelder-mead-subTPE の方が良い結果が出やすい傾向があることが分かります. ただし、関数によっては nelder-mead の方が良い結果が出ることもあります.