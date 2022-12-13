# ランダムオプティマイザのローカル実行例

ここでは，ランダムオプティマイザをローカルで実行する方法を説明します．
例として，ベンチマーク関数の一つである sphere の最適化を行います．

以下で説明するファイルは github から入手することも可能です．



## 1. ファイル構成

### config.yaml

- 最適化およびソフトウェアの設定ファイルです．


### user.py

- 与えられたパラメータからベンチマーク関数 sphere の値を計算し，aiaccel の Storage に保存するユーザプログラムです．


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
- **batch_job_timeout** - job のタイムアウト時間を設定します。[単位: 秒]

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
      lower: -3.0
      upper: 5.0
      initial: -3.0
    -
      name: "x3"
      type: "uniform_float"
      lower: -5.0
      upper: 2.7
      initial: 2.2
    -
      name: "x4"
      type: "uniform_float"
      lower: -5.0
      upper: 4.5
      initial: 4.0
    -
      name: "x5"
      type: "uniform_float"
      lower: -7.0
      upper: 6.0
```

- **search_algorithm** - 最適化アルゴリズムを設定します．この例ではランダムオプティマイザを設定しています．
- **goal** - 最適化の方向を設定します．
    - ベンチマーク関数 sphere を最小化することが目的であるため，`"minimize"` を設定しています．
- **trial_number** - 試行回数を設定します．
- **rand_seed** - 乱数の生成に使用するシードを設定します．
- **parameters** - ハイパーパラメータの各種項目を設定します．ここでは 5 次元の spehre の最適化を行うため，5 種類のパラメータを用意しています．5 つのパラメータに対して，以下の項目をそれぞれ設定する必要があります．パラメータの範囲や初期値を，全て同じにする必要はありません．
    - **name** - ハイパーパラメータの名前を設定します．
    - **type** - ハイパーパラメータのデータ型を設定します．
    - **lower / upper** - ハイパーパラメータ最小値 / 最大値を設定します．
    - **initial** - ハイパーパラメータの初期値を設定します．上の例の `"x5"` の場合のように `initial` の項目がない場合，実行時にランダムな初期値が自動で設定されます．

### user.py の作成
---

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

- numpy - ベンチマーク関数 sphere を計算するために使用します．
- aiaccel.util.aiaccel - ユーザープログラム内で定義される関数 `main()` と aiaccelとの間のインターフェイスを提供します．


#### main

```python
def main(p):
    x = np.array([p["x1"], p["x2"], p["x3"], p["x4"], p["x5"]])
    y = np.sum(x ** 2)
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
aiaccel から関数 main にハイパーパラメータを渡し，`main()` の返却値を Storage に保存します．`run` はそのインターフェイスとなるインスタンスです．メソッド `execute_and_report()` の内部で `main()` が呼ばれ，sphere の値を計算し，Storage に計算結果が保存されます．


<br>

## 3. 実行

ファイルの作成が終了したら，下記のコマンドで aiaccel を起動してください．

```bash
aiaccel-start --config config.yaml --clean
```

<br>

## 4. 最適化結果

- ハイパーパラメータ

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
        x2 = -3.0
        x3 = 2.2
        x4 = 4.0
        x5 = 

        sphere = 
        ```

    - 最適化結果
        ```
        x1 = -5.0
        x2 = -3.0
        x3 = 2.2
        x4 = 4.0
        x5 = 

        sphere = 
        ```
