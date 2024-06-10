# バジェット指定型グリッドオプティマイザのローカル環境での実行例

ここでは，バジェット指定型グリッドオプティマイザをローカルで実行する方法を説明します．
例として，ベンチマーク関数の一つである Schwefel の最適化を行います．

以下の説明では aiaccel/examples/budget_specified_grid_optimizer に保存されているファイルを編集して使用します．

## 1. ファイル構成

### config.yaml

- 最適化およびソフトウェアの設定ファイルです．
- 与えられたパラメータからベンチマーク関数 Schwefel の値を計算し，aiaccel の Storage に保存するユーザプログラムです．


<br>

## 2. ファイル作成手順

### config.yaml の作成

**generic**
```yaml
generic:
  workspace: "./work"
  job_command: "python user.py"
  batch_job_timeout: 600
  enabled_variable_name_argumentation: True
  logging_level: INFO
```
- **workspace** - aiaccel の実行に必要な一時ファイルを保存するディレクトリを指定します．
- **job_command** - ユーザープログラムを実行するためのコマンドです．
- **batch_job_timeout** - ジョブのタイムアウト時間を設定します．[単位: 秒]
- **enabled_variable_name_argumentation** - `"True"` or `"False"` によって，コマンドライン引数の指定方法が変わります．(参照： [aiaccel/examples/vlmop2/README.md](https://github.com/aistairc/aiaccel/blob/0c2559fedee384694cc7ca72d8082b8bed4dc7ad/examples/vlmop2/README.md?plain=1#L35))
- **logging_level** - ログの出力レベルを `"INFO"` に設定します．


> Windows 上で仮想環境の python プログラムを実行するために `job_command` の欄を `"path\\to\\optenv\\Scripts\\python.exe user.py"` と設定する必要があります．`"path\\to\\"` の部分はご自身の環境のパスを絶対パスで指定してください．


**resource**
```yaml
resource:
  type: "local"
  num_workers: 4
```

- **type** - 実行環境を指定します．ローカル環境で実行するためには `"local"` で設定します．
- **num_workers** - 使用するノード数を指定します．


**ABCI**

ローカル実行のため使用しません．

**optimize**
```yaml
optimize:
  search_algorithm: "aiaccel.optimizer.BudgetSpecifiedGridOptimizer"
  goal: "minimize"
  trial_number: 30
  rand_seed: 42
  grid_sampling_method: "IN_ORDER"
  grid_accept_small_trial_number: false
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
- **trial_number** - 試行回数を設定します．数値パラメータに選択肢の数が設定されていない場合，バジェット指定型グリッドオプティマイザは，この値を元に自動で選択肢数を割り当てます．
- **rand_seed** - 乱数の生成に使用するシードを設定します．
- **grid_sampling_method** - グリッド点のサンプリング方法を設定します[[参考](#付録２-グリッド点のサンプリング)]．指定可能なサンプリング方法は以下の 4 つです．
  - IN_ORDER
  - UNIFORM
  - DUPRECATABLE_RANDOM
  - RANDOM
- **grid_accept_small_trial_number** - 生成される最小のグリッド点の数よりも試行回数が小さい場合に，aiaccel の実行を強制するかを `true` または `false` で設定します[[参考](#試行回数が少ない場合の強制実行)]．
- **parameters** - ハイパパラメータの各種項目を設定します．ここでは 5 次元の Schwefel の最適化を行うため，5 種類のパラメータを用意しています．5 つのパラメータに対して，以下の項目をそれぞれ設定する必要があります．パラメータの範囲や初期値を，全て同じにする必要はありません．
    - **name** - ハイパパラメータの名前を設定します．
    - **type** - ハイパパラメータのデータ型を設定します．バジェット指定型グリッドオプティマイザでは以下の 4 つから選択することができます．
        - uniform_float - 浮動小数点数
        - uniform_int - 整数
        - categorical - カテゴリカル変数
        - ordinal - オーディナル変数
    - **lower / upper** - ハイパパラメータ最小値 / 最大値を設定します．
    - **log** -  対数スケールでパラメータ空間を分割するかを `true` または `false` で設定します．
    - **choices** - データ型が "categorical" の場合，ハイパパラメータの選択肢をリストとして設定します．
    - **sequence** - データ型が "ordinal" の場合，ハイパパラメータの選択肢をリストとして設定します．
    
> **注意**: バジェット指定型グリッドオプティマイザでは，パラメータの初期値を設定することができません．

### user.py の作成

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

aiaccel の正常終了後，最適化の結果は以下に保存されます．

- ./work/results.csv

ここで，./work はコンフィグファイルの workspace に設定したディレクトリです．

results.csv には，それぞれの試行でのパラメータの値と，そのパラメータに対する目的関数の値が保存されています．

上で実行した最適化の結果は以下のようになります．

- trial_id: 0
- parameters:
  - x1 (FLOAT) = -500.0
  - x2 (FLOAT) = 50.0
  - x3 (INT) = -500
  - x4 (CATEGORICAL) = -500
  - x5 (ORDINAL) = -500
- result: -757.799698717469


<br>

<br>

## 付録１ 通常のグリッドオプティマイザとの違い

通常のグリッドオプティマイザ (GridOptimizer) とバジェット指定型のグリッドオプティマイザ (BudgetSpecifiedGridOptimizer) では，数値パラメータ (`uniform_float` 型 および `uniform_int` 型) の選択肢の設定方法が異なります．
以下の表に数値パラメータの設定フィールドをまとめます．

|                       | GridOptimizer | BudgetSpecifiedGridOprimizer |
| --------------------: | :-----------: | :--------------------------: |
|                `name` |       ○       |              ○               |
|                `type` |       ○       |              ○               |
|               `lower` |       ○       |              ○               |
|               `upper` |       ○       |              ○               |
|                 `log` |       ○       |              ○               |
|                `step` |       ○       |              ×               |
|                `base` |       ○       |              ×               |
| `num_numeric_choices` |       ×       |              ○               |

表から分かる通り，`name`，`type`，`lower`，`upper`，および `log` といったフィールドは共通で使用されます．
一方で，通常のグリッドオプティマイザで使用される `step` と `base` はバジェット指定型のグリッドオプティマイザでは使用されません．
逆に， `num_numeric_choices` はバジェット指定型の場合にのみ使用され，通常のグリッドオプティマイザでは使用されません．

通常のグリッドオプティマイザは，数値パラメータの選択肢を値の刻み幅 `step` (`log`=True の場合は $ {\rm base^{step}} $) から生成します．
一方で，バジェット指定型の場合には，数値パラメータの選択肢は，`lower` と `upper` で指定される定義域に等間隔に (`log`=True なら対数スケール上で) 分布する `num_numeric_choices` 個の値です．

また， `step` と `base` が通常のグリッドオプティマイザで必須のフィールドであるのに対して， `num_numeric_choices` はバジェット指定型のグリッドオプティマイザであっても必須のフィールドではありません．
これは，指定されたバジェット = 試行回数 (trial_number) から適当な選択肢の数が自動で計算され，選択肢の数が設定されていないパラメータの `num_numeric_choices` として割り当てられるためです．
従って，特定のパラメータの探索回数を増やしたい (あるいは減らしたい) 場合には， `num_numeric_choices` を手動で指定すると良いでしょう．
上で説明した例では，"x3" には選択肢の数が設定されていますが， "x1" と "x2" には設定されていません．
そのため，"x1" と "x2" には自動で選択肢の数が設定されることになります．


## 付録２ グリッド点のサンプリング

### 数値パラメータの選択肢の生成
`num_numeric_choices` が手動で設定された数値パラメータ，カテゴリカルパラメータ，またはオーディナルパラメータが，それぞれ $c_i$ 個の選択肢を持つ場合，全グリッド点の探索に必要な最小の試行回数は $ m = \prod_i c_i $ と表せます．
実際にコンフィグで指定された試行回数を $n$ とすると，選択肢が指定されていない数値パラメータの選択肢は $ n' = n / m $ から計算されます．
割り当てられる選択肢の数は $ x $ 個または $ x - 1 $ 個で，選択肢を割り当てられたパラメータ同士で選択肢の数の差は 1 を超えません．
このとき，$x$ は $ x^q (x - 1)^{p - q} \geq n' $ を満たす最小の値として決定されます．ここで， $p$ は選択肢の数が未指定なパラメータの数， $q$ は $0 < q \leq p $ を満たす整数です．

例えば，試行回数が $n = 12$ のとき，2 つのパラメータ $x_1,\ x_2$ を，選択肢の数を指定せずに最適化しようとした場合，それぞれのパラメータには 4 つと 3 つの選択肢が割り当てられることになります．

自動・手動に関わらず，選択肢の数が決定した後，数値パラメータの選択肢を生成するために，aiaccel では numpy の関数を活用します．
具体的には，対数スケールでの探索が無効の場合 (`log = false`) には `numpy.linspace` で，対数スケールでの探索が有効の場合 (`log = true`) には `numpy.geomspace` でパラメータの選択肢を生成します．
このとき，指定された `lower` と `upper` はこれら 2 つの関数に，引数 _start_ および _stop_ としてそれぞれ渡されます．

ここで例として，以下のような設定を考えます．
```yaml
-
  name: x1
  type: uniform_float
  lower: 0.0
  upper: 10.0
  num_numeric_choices: 5
- 
  name: x2
  type: uniform_int
  lower: 0
  upper: 10
  num_numeric_choices: 5
- 
  name: x3
  type: uniform_int
  lower: 0
  upper: 1
  num_numeric_choices: 5
```
この場合， `x1`，`x2`，`x3` の選択肢として，以下が得られます．
```
x1 = [0.0, 2.5, 5.0, 7.5, 10.0]
x2 = [0, 2, 5, 7, 10]
x3 = [0, 1]
```

`x3` では，設定した選択肢の数 (`num_numeric_choices = 5`) よりも生成された選択肢の数 ( `= 2`) の方が少なくなっています．
これは，選択肢を生成する過程で，浮動小数点数を整数に変換する際に発生した重複を除いたためです．
具体的には，以下のような操作によって重複が省かれています．
```
x3 = [0.00, 0.25, 0.50, 0.75, 1.00]  # 浮動小数点数として選択肢を生成
   -> [0, 0, 0, 0, 1]  # 各選択肢を整数型に変換
   -> [0, 1]  # 重複する値を削除
```
`int` 型のパラメータを用いる場合，このような点にも注意して選択肢の数を設定する必要があります．


### グリッド点のサンプリング

[数値パラメータの選択肢の生成](#数値パラメータの選択肢の生成)で述べたように，自由に割り当てられる試行回数が $n'$ のとき，各パラメータの選択肢の数 $x$ と $x - 1$ は $ x^q (x - 1)^{p - q} \geq n' $ を満たすように決まります．
従って，生成されるグリッド点の数は，試行回数よりも多い場合があります．
このような場合，全てのグリッド点から一部の点を選択 (サンプリング) して探索することになります．

aiaccel では以下の 4 つのサンプリング方法を用いることができます．
- IN_ORDER
- UNIFORM
- DUPRECATABLE_RANDOM
- RANDOM


以下では，次のような設定を用いて，サンプリング方法の違いを説明します．
```yaml
optimize:
  trial_number: 10
  grid_sampling_method: "IN_ORDER"
  # grid_sampling_method: "UNIFORM"
  # grid_sampling_method: "DUPRECATABLE_RANDOM"
  # grid_sampling_method: "RANDOM"
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
この例では，試行回数が 10 回と指定されているため， 2 つのパラメータ `x1` と `x2` にはそれぞれ 4 つと 3 つの選択肢が割り当てられます [ $4 \times 3 = 12 > 10 (= {\rm trial\_number})$ ]．
従って，生成されるグリッド点は 12 点ですが，試行回数がこれより少ないため， 2 点は探索されません．


探索されるグリッド点を `x1` と `x2` が張る平面にプロットすると以下のようになります．
プロットの縦軸と横軸はそれぞれ `x1` と `x2` に対応します．
記号 `X`，`Y`，および `_` は，それぞれ 1 回だけ探索された点，2 回探索された点，および探索されなかった点を表します．
DUPRECATABLE_RANDOM および RANDOM は乱数生成に用いるシードが 42 の場合の結果です．

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

探索された`x1` と `x2` の組み合わせを表にまとめると以下のようになります．
記号の意味はプロットと同じです．

| grid_id |  x1   |  x2   | IN_ORDER | UNIFORM | DUPRECATABLE_RANDOM | RANDOM |
| ------: | :---: | :---: | :------: | :-----: | :-----------------: | :----: |
|       0 |   0   |   0   |    X     |    X    |          Y          |   X    |
|       1 |   0   |   1   |    X     |    X    |          Y          |   X    |
|       2 |   0   |   2   |    X     |    X    |          _          |   X    |
|       3 |   1   |   0   |    X     |    X    |          _          |   X    |
|       4 |   1   |   1   |    X     |    X    |          Y          |   X    |
|       5 |   1   |   2   |    X     |    _    |          _          |   _    |
|       6 |   2   |   0   |    X     |    X    |          X          |   X    |
|       7 |   2   |   1   |    X     |    X    |          _          |   X    |
|       8 |   2   |   2   |    X     |    X    |          X          |   X    |
|       9 |   3   |   0   |    X     |    X    |          _          |   X    |
|      10 |   3   |   1   |    _     |    _    |          _          |   X    |
|      11 |   3   |   2   |    _     |    X    |          Y          |   _    |

IN_ORDER，UNIFORM，および RANDOM は重複なしで探索を行います．
IN_ORDER でサンプリングした場合，無視されるグリッド点が探索空間の端に集中します．
一方で，UNIFORM または RANDOMでサンプリングした場合，無視されるグリッド点が分散するため，満遍なくパラメータを探索できます．
UNIFORM と RANDOM の違いは，サンプリングされる点が解析的に予測できるか否かです．
UNIFORM では，`numpy.linspace` を用いて探索対象となるグリッド点の id の配列をあらかじめ生成しますが，RANDOM では `numpy.random.RandomState.choice` を用いてグリッド点をランダムに選択します．

DUPRECATABLE_RANDOM でも `numpy.random.RandomState` を用いてランダムにグリッド点の選択を行いますが，RANDOM とは異なり，選択されるグリッド点に重複が発生する可能性があります．
これは，DUPRECATABLE_RANDOM でサンプリングを行う場合には，aiaccel の内部でグリッド点の "組み合わせ" を保持しないことが原因です．
例えば `x1` の選択として `[0, 1]` が，`x2` の選択肢として `[0, 1, 2]` が与えられた場合，RANDOM では `grid_points = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]` というような 6 つの組み合わせを保持しますが，DUPRECATABLE_RANDOM では `x1` と `x2` の選択肢 (計 5 つ) だけをそのまま保持します．
従って DUPRECATABLE_RANDOM は，例えば 10 個のパラメータがそれぞれ 10 個の選択肢を持つような場合に，必要なメモリの大きさの観点から RANDOM よりも優れていると言えます．

DUPRECATABLE_RANDOM と同様に，IN_ORDER および UNIFORM の場合もグリッド点を組み合わせとして保持しないため，密なグリッド空間を扱うことが可能です．

### 試行回数が少ない場合の強制実行

選択肢の数が予め決まっているパラメータを使用する場合，全グリッド点を探索するために必要な最小の試行回数が存在します．
有効な `num_numeric_choices` が設定された数値パラメータを使用した場合や，カテゴリカル型やオーディナル型のパラメータを使用した場合が，この場合に相当します．
例えば以下の設定では，9 (= 1 x 3 x 3) つのグリッド点が生成されるため，必要な最小の試行回数は 9 となります．
```yaml
optimize:
  parameters:
  -
    name: x0
    type: uniform_float
    lower: 0.0
    upper: 2.0
  -
    name: x1
    type: unifom_int
    lower: 0
    upper: 2
    num_numeric_choices: 3
  -
    name: x2
    type: categorical
    choices: ["a", "b", "c"]
```
このようなパラメータの条件を与えたにもかかわらず，試行回数が 9 に満たない場合，デフォルト設定では aiaccel は最適化を行わず強制終了します．
もし，必要最小の試行回数よりも小さな試行回数で探索を行いたい場合には，optimize の下の階層で `grid_accept_small_trial_number` を `true` に設定します．
```yaml
optimize:
  grid_accept_small_trial_number: true
```
この設定で aiaccel を実行すると，上の例では 9 つのグリッド点から適当な点 (例えば trial_number = 5 なら 5 つの点) がサンプリングされます．

