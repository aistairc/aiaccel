# バジェット指定型グリッドオプティマイザのローカル環境での実行例

ここでは，バジェット指定型グリッドオプティマイザをローカルで実行する方法を説明します．
例として，ベンチマーク関数の一つである Schwefel の最適化を行います．

以下の説明では aiaccel/examples/budget_specified_grid に保存されているファイルを編集して使用します．

## 1. 生成したグリッド点を順番にサンプリングする場合

### 1. ファイル構成

ファイルは aiaccel/examples/budget_specified_grid/1_sampling_in_order に保存されています．

#### config.yaml

- 最適化およびソフトウェアの設定ファイルです．

#### user.py

- 与えられたパラメータからベンチマーク関数 Schwefel の値を計算し，aiaccel の Storage に保存するユーザプログラムです．


<br>

### 2. ファイル作成手順

#### config.yaml の作成

**generic**
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

**resource**
```yaml
resource:
  type: "local"
  num_node: 4
```

- **type** - 実行環境を指定します．ローカル環境で実行するためには `"local"` で設定します．
- **num_node** - 使用するノード数を指定します．


**ABCI**

ローカル実行なので使用しません．

**optimize**
```yaml
optimize:
  search_algorithm: 'aiaccel.optimizer.GridOptimizer'
  goal: "minimize"
  trial_number: 30
  rand_seed: 42
  parameters:
    -
      name: "x1"
      type: "uniform_float"
      lower: -500.0
      upper: 500.0
    -
      name: "x2"
      type: "uniform_float"
      lower: 50.0
      upper: 500.0
      log: true
    -
      name: "x3"
      type: "uniform_int"
      lower: -500
      upper: 500
      num_grid_points: 3
    -
      name: "x4"
      type: "categorical"
      choices: [-500, 0, 500]
    -
      name: "x5"
      type: "ordinal"
      sequence: [-500, 0, 500]
```

- **search_algorithm** - 最適化アルゴリズムを設定します．この例ではバジェット指定型グリッドオプティマイザを設定しています．
- **goal** - 最適化の方向を設定します．
    - 関数 Schwefel を最小化することが目的であるため，`"minimize"` を設定しています．
- **trial_number** - 試行回数を設定します．
- **rand_seed** - 乱数の生成に使用するシードを設定します．
- **parameters** - ハイパパラメータの各種項目を設定します．ここでは 5 次元の Schwefel の最適化を行うため，5 種類のパラメータを用意しています．5 つのパラメータに対して，以下の項目をそれぞれ設定する必要があります．パラメータの範囲や初期値を，全て同じにする必要はありません．
    - **name** - ハイパパラメータの名前を設定します．
    - **type** - ハイパパラメータのデータ型を設定します．ここでは例として `"uniform_float"` に設定していますが，グリッドオプティマイザでは，以下の 4 つから選択することができます．
        - uniform_float - 浮動小数点数
        - uniform_int - 整数
        - categorical - カテゴリカル変数
        - ordinal - オーディナル変数
    - **lower / upper** - ハイパパラメータ最小値 / 最大値を設定します．
    - **log** -  対数スケールでパラメータ空間を分割するかを `true` または `false` で設定します．
    
**注意**: バジェット指定型グリッドオプティマイザでは，パラメータの初期値を設定することができません．

#### user.py の作成

`user.py` は以下のように記述します．
```python
import numpy as np
from aiaccel.util.aiaccel import Run


def main(p):
    x = np.array([p["x1"], p["x2"], p["x3"], p["x4"], p["x5"]])
    y = -np.sum(x * np.sin(np.sqrt(np.abs(x))))
    return float(y)


if __name__ == "__main__":
    run = Run()
    run.execute_and_report(main)

```

**モジュール**
```python
import numpy as np
from aiaccel.util.aiaccel import Run
```

必要なモジュールをインポートします．

- numpy - 関数 Schwefel を計算するために使用します．
- aiaccel.util.aiaccel - ユーザープログラム内で定義される関数 `main()` と aiaccelとの間のインターフェイスを提供します．


**main**
```python
def main(p):
    x = np.array([p["x1"], p["x2"], p["x3"], p["x4"], p["x5"]])
    y = -np.sum(x * np.sin(np.sqrt(np.abs(x))))
    return float(y)
```
最適化対象の関数で，aiaccel はこの関数の `return` 値を最小化します．
引数にハイパパラメータの辞書型オブジェクトを取ります，
この例では，関数 Schwefel の値を計算し，返却します．

**実行部分**
```python
if __name__ == "__main__":
    run = Run()
    run.execute_and_report(main)
```
aiaccel から関数 `main()` にハイパパラメータを渡し，`main()` の返却値を Storage に保存します．
`run` はそのインターフェイスとなるインスタンスです．
メソッド `execute_and_report()` の内部で `main()` が値を計算し，Storage に計算結果が保存されます．


<br>

### 3. 実行

作成した config.yaml と user.py が保存されているディレクトリに移動し，下記のコマンドで aiaccel を起動してください．

```console
> aiaccel-start --config config.yaml --clean
```

- コマンドラインオプション引数
    - `--config` - 設定ファイルを読み込むためのオプション引数です．読み込むコンフィグのパスを記述します．
    - `--clean` - aiaccel の起動ディレクトリ内に config.yaml の workspace で指定したディレクトリが存在する場合，削除してから実行するためのオプション引数です．

<br>

### 4. 結果の確認

aiaccel の正常終了後，最適化の結果は以下の 2 か所に保存されます．

- ./work/results.csv
- ./work/result/{trial_id}.hp

ここで，./work はコンフィグファイルの workspace に設定したディレクトリです．

results.csv には，それぞれの試行でのパラメータの値と，そのパラメータに対する目的関数の値が保存されています．
result/{trial_id}.hp は，{trial_id} 回目の試行のパラメータと関数の値が YAML 形式で保存されています．
さらに，同じフォルダには final_result.result というファイルが作成され，全試行中で最良のパラメータと目的関数の値が YAML 形式で保存されます．

上で実行した最適化の結果は以下のようになります．

- trial_id: 0
- parameters:
  - x1 (FLOAT) = -500.0
  - x2 (FLOAT) = 50.0
  - x3 (INT) = -500
  - x4 (CATEGORICAL) = -500
  - x5 (ORDINAL) = -500
- result: -757.799698717469


## グリッド点のサンプリング

### グリッド点の生成
試行回数 $n$ からグリッド点の位置を決定するため，選択肢の数が指定されていない `float` 型または `int` 型のパラメータには $ x $ 個または $ x - 1 $ 個の選択肢が割り当てられます．`num_grid_point` が指定された `float` や `int`，或いはカテゴリカルやオーディナルな変数がそれぞれ $c_i$ 個の選択肢をあらかじめ与えられていた場合は，サイズが $ m = \prod c_i $ の探索空間を除いた $ n' = n / m $ を選択肢の数が未指定のパラメータに割り当てます．このとき，$x$ は $ x^q (x - 1)^{p - q} \geq n' $ を満たす最小の値として決定されます．ここで， $p$ は選択肢の数が未指定なパラメータの数， $q$ は $0 < q \leq p $ を満たす整数です．

例えば，試行回数が $n = 12$ のとき，2 つのパラメータ $x_1,\ x_2$ を，グリッド点数を指定せずに最適化しようとした場合，それぞれのパラメータは 4 つと 3 つの選択肢から選択されることになります．

自動・手動に関わらず，選択肢の数が決定した後，パラメータの候補を生成するために，aiaccel では numpy の関数を活用します．
具体的には，対数スケールでの探索が無効の場合 (`log = false`)，`numpy.linspace` で，対数スケールでの探索が友好の場合 (`log = true`)，`numpy.geomspace` でパラメータの選択肢を生成します．
このとき，指定された `lower` と `upper` はこれら 2 つの関数の引数 _start_ および _stop_ にそれぞれ渡されます．
また，`int` 型のパラメータの場合には，これらの関数の引数 _dtype_ に `int` を指定します．

例として，以下のような設定を考えると，
```yaml
-
  name: x1
  type: uniform_float
  lower: 0.0
  upper: 10.0
  num_grid_points: 5
- 
  name: x2
  type: uniform_int
  lower: 0
  upper: 10
  num_grid_points: 5
- 
  name: x3
  type: uniform_int
  lower: 0
  upper: 1
  num_grid_points: 5
```
`x1`，`x2`，`x3` の選択肢は，
```
x1 = [ 0.0,  2.5,  5.0,  7.5, 10.0]
x2 = [ 0,  2,  5,  7, 10]
x3 = [0, 1]
```
となります．

`x3` では，設定した選択肢の数 (`num_grid_points = 5`) よりも生成された選択肢の数 ( `= 2`) の方が少ないです．
これは，選択肢を生成する過程で，浮動小数点数を整数に変換する際に発生した重複を除いたためです．
このように，`int` 型のパラメータでは選択肢の数に注意する必要があります．


### グリッド点のサンプリング

[グリッド点の生成](#グリッド点の生成)で述べたように，自由に割り当てられる試行回数が $n'$ のとき，各パラメータの選択肢の数 $x$ と $x - 1$ は $ x^q (x - 1)^{p - q} \geq n' $ を満たすように決まります．
従って，生成されるグリッド点の数は，試行回数よりも多い場合があります．
このような場合，全てのグリッド点から一部を選択 (サンプリング) して探索することになります．

aiaccel では 4 つのサンプリング手法を用いることができます．
即ち
- IN_ORDER
- UNIFORM
- DUPRECATABLE_RANDOM
- RANDOM

の 4 つです．

以下では，例として，次のような設定の場合に，どのグリッド点を探索することになるかを説明します．
```yaml
optimize:
  trial_number: 10
  grid_sampling_method: "IN_ORDER"
  # "UNIFORM", "DUPRECATABLE_RANDOM", or "RANDOM"
  parameters:
    -
      name: "x1"
      type: "uniform_float"
      lower: 0
      upper: 3
    -
      name: "x2"
      type: "uniform_float"
      lower: 0
      upper: 2
```
この例では，試行回数が 10 回と指定されているため， 2 つのパラメータ `x1` と `x2` にはそれぞれ 4 つと 3 つの選択肢が割り当てられます ( $4 \times 3 = 12 > 10 (= {\rm trial\_number})$ )．
従って，生成されるグリッド点は 12 点になりますが，試行回数がこれより少ないため， 2 点は探索されないことになります．


探索されるグリッド点を `x1` と `x2` が張る平面にプロットすると以下のようになります．
プロットの縦軸と横軸はそれぞれ `x1` と `x2` に対応します．
また，記号 `X`，`Y`，および `_` はそれぞれ 1 回だけ探索された点，2 回探索された点，および探索されなかった点を表します．
DUPRECATABLE_RANDOM および RANDOM は seed = 42 の場合の結果です．

*IN_ORDER*
```
     x2
      0  1  2 
x1 0  X  X  X
   1  X  X  X 
   2  X  X  X    X: searched
   3  X  _  _    _: ignored
```

*UNIFORM*
```
     x2
      0  1  2 
x1 0  X  X  X
   1  X  X  _ 
   2  X  X  X    X: searched
   3  X  _  X    _: ignored
```

*DUPRICATABLE_RANDOM*
```
     x2          (seed = 42)
      0  1  2 
x1 0  Y  Y  _
   1  _  Y  _    X: searched
   2  X  _  X    Y: searched twice
   3  _  _  Y    _: ignored
```

*RANDOM*
```
     x2          (seed = 42)
      0  1  2 
x1 0  X  X  X
   1  X  X  _ 
   2  X  X  X    X: searched
   3  X  X  _    _: ignored
```

IN_ORDER，UNIFORM，および RANDOM は重複なしで探索を行います．
IN_ORDER でサンプリングした場合，無視されるグリッド点が探索空間の端に集中します．
一方で，UNIFORM または RANDOMでサンプリングした場合，無視されるグリッド点が分散するため，満遍なくパラメータを探索できます．
UNIFORM と RANDOM の違いは，解析的にサンプリングされる点が予測できるか否かです．
UNIFORM では，`numpy.linspace` を用いて探索対象となるグリッド点の id の配列を生成していますが，RANDOM では `numpy.random.RandomState.choice` を用いてグリッド点を選択します．

DUPRECATABLE_RANDOM では，RANDOM と同様に `numpy.random.RandomState` を用いてグリッド点の選択を行いますが，重複が発生する可能性があります．
これは，DUPRECATABLE_RANDOM でサンプリングを行う場合には，aiaccel の内部でグリッド点の "組み合わせ" を保持しないことが原因です．
例えば `x1` の選択として `[0, 1]` が，`x2` の選択肢として `[0, 1, 2]` が与えられた場合，RANDOM では `grid_points = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]` というような 6 つの組み合わせを保持しますが，DUPRECATABLE_RANDOM では `x1` と `x2` の選択肢 (計 5 つ) をそのまま保持します．
従って DUPRECATABLE_RANDOM は，例えば 10 個のパラメータがそれぞれ 10 個の選択肢を持つような場合に，メモリの大きさの観点から RANDOM に対して大きな優位性を持ちます．

DUPRECATABLE_RANDOM と同様に，IN_ORDER および UNIFORM の場合もグリッド点を組み合わせとして保持しないため，メモリ上で大きな探索空間を扱うことが可能です．

| grid_id | x1  | x2  | IN_ORDER | UNIFORM | DUPRECATABLE_RANDOM | RANDOM | 
| -: | :-: | :-: | :-: | :-: | :-: | :-: |
|  0 |  0  |  0  |  X  |  X  |  Y  |  X  |
|  1 |  0  |  1  |  X  |  X  |  Y  |  X  |
|  2 |  0  |  2  |  X  |  X  |  _  |  X  |
|  3 |  1  |  0  |  X  |  X  |  _  |  X  |
|  4 |  1  |  1  |  X  |  X  |  Y  |  X  |
|  5 |  1  |  2  |  X  |  _  |  _  |  _  |
|  6 |  2  |  0  |  X  |  X  |  X  |  X  |
|  7 |  2  |  1  |  X  |  X  |  _  |  X  |
|  8 |  2  |  2  |  X  |  X  |  X  |  X  |
|  9 |  3  |  0  |  X  |  X  |  _  |  X  |
| 10 |  3  |  1  |  _  |  _  |  _  |  X  |
| 11 |  3  |  2  |  _  |  X  |  Y  |  _  |
