# TPE オプティマイザのローカル環境 (python_local モード) での実行例

ここでは，TPE オプティマイザを `python_local` モードを用いてローカルで実行する方法を説明します．
例として，ベンチマーク関数の一つである Styblinski-Tang の最適化を行います．

以下の説明では aiaccel/examples/styblinski-tang に保存されているファイルを編集して使用します．


## 1. ファイル構成

### config.yaml

- 最適化およびソフトウェアの設定ファイルです．

### user.py

- 与えられたパラメータからベンチマーク関数 Styblinski-Tang の値を計算し，aiaccel の Storage に保存するユーザプログラムです．


<br>

## 2. ファイル作成手順

### config.yaml の作成


#### generic
```yaml
generic:
  workspace: "./work"
  job_command: "python user.py"
  python_file: "./user.py"
  function: "main"
batch_job_timeout: 600
```
- **workspace** - aiaccel の実行に必要な一時ファイルを保存するディレクトリを設定します．
- **job_command** - ユーザープログラムを実行するためのコマンドです．`python_local` モードでは使用されませんが，実行時に読み込むため，記述します．
- **python_file** - python で実装された最適化対象の関数のファイルパスを設定します．
- **function** - 最適化対象の関数名を設定します．
- **batch_job_timeout** - ジョブのタイムアウト時間を設定します．[単位: 秒]

#### resource
```yaml
resource:
  type: "python_local"
  num_workers: 4
```

- **type** - 実行環境を指定します．`python_local` モードを使用してローカルで実行するためには `"python_local"` と設定します．
- **num_workers** - 使用するノード数を指定します．


#### optimize
```yaml
optimize:
  search_algorithm: 'aiaccel.optimizer.TpeOptimizer'
  goal: "minimize"
  trial_number: 30
  rand_seed: 42
  parameters:
    -
      name: "x1"
      type: "uniform_float"
      lower: -5.0
      upper: 5.0
      initial: 0.0
    -
      name: "x2"
      type: "uniform_float"
      lower: -5.0
      upper: 5.0
      initial: 0.0
    -
      name: "x3"
      type: "uniform_float"
      lower: -5.0
      upper: 5.0
      initial: 0.0
    -
      name: "x4"
      type: "uniform_float"
      lower: -5.0
      upper: 5.0
      initial: 0.0
    -
      name: "x5"
      type: "uniform_float"
      lower: -5.0
      upper: 5.0
      initial: 0.0
```

- **search_algorithm** - 最適化アルゴリズムを設定します．この例では TPE オプティマイザを設定しています．
- **goal** - 最適化の方向を設定します．
    - ベンチマーク関数 Styblinski-Tang を最小化することが目的であるため，`"minimize"` を設定しています．
- **trial_number** - 試行回数を設定します．
- **rand_seed** - 乱数の生成に使用するシードを設定します．
- **parameters** - ハイパパラメータの各種項目を設定します．ここでは 5 次元の Styblinski-Tang の最適化を行うため，5 種類のパラメータを用意しています．5 つのパラメータに対して，以下の項目をそれぞれ設定する必要があります．パラメータの範囲や初期値を，全て同じにする必要はありません．
    - **name** - ハイパパラメータの名前を設定します．
    - **type** - ハイパパラメータのデータ型を設定します．ここでは例として `"uniform_float"` に設定していますが，TPE オプティマイザでは，以下の 3 つのタイプから選択することができます．
        - uniform_float - 浮動小数点数
        - uniform_int - 整数
        - categorical - カテゴリカル変数
    - **lower / upper** - ハイパパラメータ最小値 / 最大値を設定します．
    - **log** -  対数スケールでパラメータ空間を分割するかを `true` または `false` で設定します．
    - **initial** - ハイパパラメータの初期値を設定します．

### user.py の作成

`user.py` は以下のように記述します．
```python
import numpy as np


def main(p):
    x = np.array([p["x1"], p["x2"], p["x3"], p["x4"], p["x5"]])
    t1 = np.sum(x ** 4)
    t2 = -16 * np.sum(x ** 2)
    t3 = 5 * np.sum(x)
    y = 0.5 * (t1 + t2 + t3)
    return float(y)

```

#### モジュール

```python
import numpy as np
```

必要なモジュールをインポートします．

- numpy - 関数 Styblinski-Tang を計算するために使用します．


#### main

```python
def main(p):
    x = np.array([p["x1"], p["x2"], p["x3"], p["x4"], p["x5"]])
    t1 = np.sum(x ** 4)
    t2 = -16 * np.sum(x ** 2)
    t3 = 5 * np.sum(x)
    y = 0.5 * (t1 + t2 + t3)
    return float(y)
```
最適化対象の関数で，aiaccel はこの関数の `return` 値を最小化します．
引数にハイパパラメータの辞書型オブジェクトを取ります．
この例では，与えられたパラメータに対してベンチマーク関数 Styblinski-Tang の値を計算し，返却します．

#### 注意
`python_local` モードで実行する場合，aiaccel は user.py から `main()` をインポートして使用します．
そのため，`local` や `abci` の場合のように，user.py の内部で `aiaccel.util.aiaccel.Run` などを用いて関数の実行と Storage への結果の保存を行う必要はありません．


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

    - Styblinski-Tang

- 最適化手法
    - TPE

- 結果比較

    - デフォルトパラメータ
        ```
        x1 = 0.0
        x2 = 0.0
        x3 = 0.0
        x4 = 0.0
        x5 = 0.0

        result = 0.0
        ```

    - 最適化結果
        ```
        x1 = -3.930303675338723
        x2 =  2.640453540706446
        x3 = -2.954080362853035
        x4 = -2.359296335376299
        x5 =  2.690131733893075
        
        result = -138.00660281690844
        ```
