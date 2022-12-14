# グリッドオプティマイザのローカル実行例

ここでは，グリッドオプティマイザをローカルで実行する方法を説明します．
例として，ベンチマーク関数の一つである Schwefel の最適化を行います．

以下の説明では aiaccel/examples/schwefel に保存されているファイルを編集して使用します．


## 1. ファイル構成


### config.yaml

- 最適化およびソフトウェアの設定ファイルです．
- aiaccel/examples/schwefel に存在するファイルは，デフォルトでは Nelder-Mead 法を用いた最適化をローカルで実行する設定になっています．

### user.py

- 与えられたパラメータからベンチマーク関数 Schwefel の値を計算し，aiaccel の Storage に保存するユーザプログラムです．


<br>

## 2. ファイル作成手順

### config.yaml の作成
---

#### generic
```yaml
generic:
  workspace: "./work"
  job_command: "python user.py"
  batch_job_timeout: 600
```
- **workspace** - aiaccel の実行に必要な一時ファイルを保存するディレクトリを指定します．
- **job_command** - ユーザープログラムを実行するためのコマンドです．
- **batch_job_timeout** - job のタイムアウト時間を設定します．[単位: 秒]

#### resource
```yaml
resource:
  type: "local"
  num_node: 4
```

- **type** - 実行環境を指定します。ローカル環境で実行するためには `"local"` で設定します．
- **num_node** - 使用するノード数を指定します．


#### optimize
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
      step: 10
      log: false
      base: 10
    -
      name: "x2"
      type: "uniform_float"
      lower: -500.0
      upper: 500.0
      step: 10
      log: false
      base: 10
    -
      name: "x3"
      type: "uniform_float"
      lower: -500.0
      upper: 500.0
      step: 10
      log: false
      base: 10
    -
      name: "x4"
      type: "uniform_float"
      lower: -500.0
      upper: 500.0
      step: 10
      log: false
      base: 10
    -
      name: "x5"
      type: "uniform_float"
      lower: -500.0
      upper: 500.0
      step: 10
      log: false
      base: 10

```

- **search_algorithm** - 最適化アルゴリズムを設定します．この例ではグリッドオプティマイザを設定しています．
- **goal** - 最適化の方向を設定します．
    - 関数 Schwefel を最小化することが目的であるため，`"minimize"` を設定しています．
- **trial_number** - 試行回数を設定します．
- **rand_seed** - 乱数の生成に使用するシードを設定します．
- **parameters** - ハイパーパラメータの各種項目を設定します．ここでは 5 次元の Schwefel の最適化を行うため，5 種類のパラメータを用意しています．5 つのパラメータに対して，以下の項目をそれぞれ設定する必要があります．パラメータの範囲や初期値を，全て同じにする必要はありません．
    - **name** - ハイパーパラメータの名前を設定します．
    - **type** - ハイパーパラメータのデータ型を設定します．ここでは例として `"uniform_float"` に設定していますが，グリッドオプティマイザでは，以下の 4 つから選択することができます．
        - uniform_float - 浮動小数点数
        - uniform_int - 整数
        - categorical - カテゴリカル変数
        - ordinal - オーディナル変数
    - **lower / upper** - ハイパーパラメータ最小値 / 最大値を設定します．
    - **step** - パラメータ空間を分割するステップサイズを設定します．
    - **log** -  対数スケールでパラメータ空間を分割するかを `true` または `false` で設定します．
    - **base** - パラメータ空間を対数スケールで分割する際に使用する基数を設定します．

注意: グリッドオプティマイザを使用する際は，パラメータの初期値を設定することができません．

### user.py の作成
---

`user.py` は以下のように記述します．
```python
import numpy as np
from aiaccel.util import aiaccel


def main(p):
    x = np.array([p["x1"], p["x2"], p["x3"], p["x4"], p["x5"]])
    y = -np.sum(x * np.sin(np.sqrt(np.abs(x))))
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

- numpy - 関数 Schwefel を計算するために使用します．
- aiaccel.util.aiaccel - ユーザープログラム内で定義される関数 `main()` と aiaccelとの間のインターフェイスを提供します．


#### main

```python
def main(p):
    x = np.array([p["x1"], p["x2"], p["x3"], p["x4"], p["x5"]])
    y = -np.sum(x * np.sin(np.sqrt(np.abs(x))))
    return float(y)
```
最適化対象のメイン関数で，aiaccel はこの関数の `return` 値を最小化します．
引数にハイパーパラメータの辞書型オブジェクトを取り，ハイパーハイパーパラメータの二乗和を返却します．

#### 実行部分
```python
if __name__ == "__main__":
    run = aiaccel.Run()
    run.execute_and_report(main)
```
aiaccel から関数 main にハイパーパラメータを渡し，`main()` の返却値を Storage に保存します．`run` はそのインターフェイスとなるインスタンスです．メソッド `execute_and_report()` の内部で `main()` が値を計算し，Storage に計算結果が保存されます．


<br>

## 3. 実行

作成した config.yaml と user.py が保存されているディレクトリに移動し，下記のコマンドで aiaccel を起動してください．

```console
> aiaccel-start --config config.yaml --clean
```

- コマンドラインオプション引数
    - `--config` - 設定ファイルを読み込むためのオプション引数です．読み込むコンフィグのパスを記述します．
    - `--clean` - aiaccel の起動ディレクトリ内に config.yaml の `workspace` で指定したディレクトリが存在する場合，削除してから実行するためのオプション引数です．

<br>

## 4. 最適化結果

- ハイパーパラメータ

    - x1
    - x2
    - x3
    - x4
    - x5

- 評価値

    - Schwefel

- 最適化手法

    - Grid

- 結果比較

    - 最適化結果
    
        ```
        x1 = -500.0
        x2 = -500.0
        x3 = -500.0
        x4 = -500.0
        x5 = -300.0

        results = -1022.0952317469887
        ```
