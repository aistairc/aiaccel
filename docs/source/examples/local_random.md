# ランダムオプティマイザのローカル環境での実行例

ここでは，ランダムオプティマイザをローカルで実行する方法を説明します．
例として，ベンチマーク関数の一つである sphere の最適化を行います．

以下の説明では aiaccel/examples/sphere に保存されているファイルを編集して使用します．



## 1. ファイル構成

### config.yaml

- 最適化およびソフトウェアの設定ファイルです．

### user.py

- 与えられたパラメータからベンチマーク関数 sphere の値を計算し，aiaccel の Storage に保存するユーザプログラムです．


<br>

## 2. ファイル作成手順

### config.yaml の作成

#### generic
```yaml
generic:
  workspace: "./work"
  job_command: "python user.py"
  batch_job_timeout: 600
```
- **workspace** - aiaccel の実行に必要な一時ファイルを保存するディレクトリを指定します．
- **job_command** - ユーザープログラムを実行するためのコマンドです．
- **batch_job_timeout** - ジョブのタイムアウト時間を設定します．[単位: 秒]

```{note}
Windows では，仮想環境の python で実行するためには `job_command` の欄を `"optenv/Scripts/python.exe"` のように設定する必要があります．
```

#### resource
```yaml
resource:
  type: "local"
  num_workers: 4
```

- **type** - 実行環境を指定します．ローカル環境で実行するためには `"local"` で設定します．
- **num_workers** - 使用するノード数を指定します．


#### optimize
```yaml
optimize:
  search_algorithm: 'aiaccel.optimizer.RandomOptimizer'
  goal: "minimize"
  trial_number: 100
  rand_seed: 42
  parameters:
    -
      name: "x1"
      type: "uniform_float"
      lower: -5.0
      upper: 5.0
      initial: -5.0
    -
      name: "x2"
      type: "uniform_float"
      lower: -5.0
      upper: 5.0
      initial: -5.0
    -
      name: "x3"
      type: "uniform_float"
      lower: -5.0
      upper: 5.0
      initial: -5.0
    -
      name: "x4"
      type: "uniform_float"
      lower: -5.0
      upper: 5.0
      initial: -5.0
    -
      name: "x5"
      type: "uniform_float"
      lower: -5.0
      upper: 5.0
```

- **search_algorithm** - 最適化アルゴリズムを設定します．この例ではランダムオプティマイザを設定しています．
- **goal** - 最適化の方向を設定します．
    - 関数 sphere を最小化することが目的であるため，`"minimize"` を設定しています．
- **trial_number** - 試行回数を設定します．
- **rand_seed** - 乱数の生成に使用するシードを設定します．
- **parameters** - ハイパパラメータの各種項目を設定します．ここでは 5 次元の spehre の最適化を行うため，5 種類のパラメータを用意しています．5 つのパラメータに対して，以下の項目をそれぞれ設定する必要があります．パラメータの範囲や初期値を，全て同じにする必要はありません．
    - **name** - ハイパパラメータの名前を設定します．
    - **type** - ハイパパラメータのデータ型を設定します．ここでは例として `"uniform_float"` に設定していますが，ランダムオプティマイザでは，以下の 4 つから選択することができます．
        - uniform_float - 浮動小数点数
        - uniform_int - 整数
        - categorical - カテゴリカル変数
        - ordinal - オーディナル変数
    - **lower / upper** - ハイパパラメータ最小値 / 最大値を設定します．
    - **log** -  対数スケールでパラメータ空間を分割するかを `true` または `false` で設定します．
    - **initial** - ハイパパラメータの初期値を設定します．上の例の `"x5"` の場合のように `initial` の項目がない場合，実行時にランダムな初期値が自動で設定されます．

### user.py の作成

`user.py` は以下のように記述します．
```python
import numpy as np
from aiaccel.util import aiaccel


def main(p):
    x = np.array([p["x1"], p["x2"], p["x3"], p["x4"], p["x5"]])
    y = np.sum(x ** 2)
    return float(y)


if __name__ == "__main__":
    run = aiaccel.Run()
    run.execute_and_report(main)

```

#### モジュール

```python
import numpy as np
from aiaccel.util import aiaccel
```

必要なモジュールをインポートします．

- numpy - 関数 sphere を計算するために使用します．
- aiaccel.util.aiaccel - ユーザープログラム内で定義される関数 `main()` と aiaccelとの間のインターフェイスを提供します．


#### main

```python
def main(p):
    x = np.array([p["x1"], p["x2"], p["x3"], p["x4"], p["x5"]])
    y = np.sum(x ** 2)
    return float(y)
```
最適化対象の関数で，aiaccel はこの関数の `return` 値を最小化します．
引数にハイパパラメータの辞書型オブジェクトを取り，ハイパパラメータの二乗和を返却します．

#### 実行部分
```python
if __name__ == "__main__":
    run = aiaccel.Run()
    run.execute_and_report(main)
```
aiaccel から関数 `main()` にハイパパラメータを渡し，`main()` の返却値を Storage に保存します．
`run` はそのインターフェイスとなるインスタンスです．
メソッド `execute_and_report()` の内部で `main()` が値を計算し，Storage に計算結果が保存されます．


<br>

## 3. 実行

作成した config.yaml と user.py が保存されているディレクトリに移動し，下記のコマンドで aiaccel を起動してください．

```console
> aiaccel-start --config config.yaml --clean
```

- コマンドラインオプション引数
    - `--config` - 設定ファイルを読み込むためのオプション引数です．読み込むコンフィグのパスを記述します．
    - `--clean` - aiaccel の起動ディレクトリ内に config.yaml の workspace で指定したディレクトリが存在する場合，削除してから実行するためのオプション引数です．

<br>

## 4. 結果の確認

aiaccel の正常終了後，最適化の結果は以下の 2 か所に保存されます．

- ./work/results.csv
- ./work/result/{trial_id}.hp

ここで，./work はコンフィグファイルの workspace に設定したディレクトリです．

results.csv には，それぞれの試行でのパラメータの値と，そのパラメータに対する目的関数の値が保存されています．
result/{trial_id}.hp は，{trial_id} 回目の試行のパラメータと関数の値が YAML 形式で保存されています．
さらに，同じフォルダには final_result.result というファイルが作成され，全試行中で最良のパラメータと目的関数の値が YAML 形式で保存されます．

上で実行した最適化の結果は以下のようになります．

- ハイパパラメータ

    - x1
    - x2
    - x3
    - x4
    - x5

- 評価値

    - sphere

- 最適化手法
    - Random

- 結果比較

    - デフォルトパラメータ
        ```
        x1 = -5.0
        x2 = -5.0
        x3 = -5.0
        x4 = -5.0
        x5 = -1.254598811526375 (自動生成)

        sphere = 101.57401817788339
        ```

    - 最適化結果
        ```
        x1 = -1.9575775704046228
        x2 =  0.24756431632237863
        x3 = -0.6805498135788426
        x4 = -2.0877085980195806
        x5 =  1.118528947223795

        sphere = 9.966180279652084
        ```
