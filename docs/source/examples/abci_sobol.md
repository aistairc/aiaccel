# ソボルオプティマイザの ABCI 環境での実行例

ここでは，ソボルオプティマイザをABCI環境で実行する方法を説明します．例として，以下 user.py の main 内で定義されている多項式をベンチマーク関数として最適化を行います．

以下の説明では aiaccel/examples/benchmark に保存されているファイルを編集して使用します．


## 1. ファイル構成

### config.yaml

- 最適化およびソフトウェアの設定ファイルです．
- aiaccel/examples/benchmark に存在するファイルは，デフォルトでは Nelder-Mead 法を用いた最適化をローカルで実行する設定になっています．

### user.py

- 与えられたパラメータからベンチマーク関数の値を計算し，aiaccel の Storage に保存するユーザプログラムです．


### job_script_preamble.sh

- ABCIで使用するモジュール指定やジョブ設定を行うためのシェルスクリプトファイルです．

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
  type: "abci"
  num_node: 4
```

- **type** - 実行環境を指定します。ABCI環境で実行するためには `"abci"` で設定します．
- **num_node** - 使用するノード数を指定します．


#### ABCI
```yaml
ABCI:
  group: "[group]"
  job_script_preamble: "./job_script_preamble.sh"
  job_execution_options: ""

```

- **group** - 所属しているABCIグループを指定します．
- **job_script_preamble** - ABCIの設定を記述したシェルスクリプトのファイルを指定します．


#### optimize
```yaml
optimize:
  search_algorithm: 'aiaccel.optimizer.SobolOptimizer'
  goal: "minimize"
  trial_number: 100
  rand_seed: 42
  parameters:
    -
      name: "x1"
      type: "uniform_float"
      lower: 0.0
      upper: 5.0
    -
      name: "x2"
      type: "uniform_float"
      lower: 0.0
      upper: 5.0
```

- **search_algorithm** - 最適化アルゴリズムを設定します．この例ではソボルオプティマイザを設定しています．
- **goal** - 最適化の方向を設定します．
    - ベンチマーク関数を最小化することが目的であるため，`"minimize"` を設定しています．
- **trial_number** - 試行回数を設定します．
- **rand_seed** - 乱数の生成に使用するシードを設定します．
- **parameters** - ハイパーパラメータの各種項目を設定します．ここでは 2 次元のベンチマーク関数の最適化を行うため，2 種類のパラメータを用意しています．2 つのパラメータに対して，以下の項目をそれぞれ設定する必要があります．パラメータの範囲や初期値を，全て同じにする必要はありません．
    - **name** - ハイパーパラメータの名前を設定します．
    - **type** - ハイパーパラメータのデータ型を設定します．
    - **lower / upper** - ハイパーパラメータ最小値 / 最大値を設定します．
    - **initial** - SobolOptimizer では，initial値は必要ありません．

### user.py の作成
---

`user.py` は以下のように記述します．
```python
from aiaccel.util import aiaccel


def main(p):
    y = (p["x1"]**2) - (4.0 * p["x1"]) + (p["x2"]**2) - p["x2"] - (p["x1"] * p["x2"])

    return float(y)


if __name__ == "__main__":
    run = aiaccel.Run()
    run.execute_and_report(main)

```

#### モジュール

```python
from aiaccel.util import aiaccel
```

必要なモジュールをインポートします．

- aiaccel.util.aiaccel - ユーザープログラム内で定義される関数 `main()` と aiaccelとの間のインターフェイスを提供します．


#### main

```python
def main(p):
    y = (p["x1"]**2) - (4.0 * p["x1"]) + (p["x2"]**2) - p["x2"] - (p["x1"] * p["x2"])

    return float(y)

```
最適化対象のメイン関数で，aiaccel はこの関数の `return` 値を最小化します．
引数にハイパーパラメータの辞書型オブジェクトを取り，多項式の計算結果を返却します．

#### 実行部分
```python
if __name__ == "__main__":
    run = aiaccel.Run()
    run.execute_and_report(main)
```
aiaccel から関数 main にハイパーパラメータを渡し，`main()` の返却値を Storage に保存します．`run` はそのインターフェイスとなるインスタンスです．メソッド `execute_and_report()` の内部で `main()` が呼ばれ，目的関数の値を計算し，Storage に計算結果が保存されます．


### job_script_preamble.shの作成
---
`job_script_preamble.sh` は、ABCIにジョブを投入するためのバッチファイルのベースファイルです．
このファイルには事前設定を記述します．
ここに記述した設定が全てのジョブに適用されます．

```bash
#!/bin/bash

#$-l rt_C.small=1
#$-j y
#$-cwd
```

- ABCIのバッチジョブ実行オプションを指定しています．
    - 参考: https://docs.abci.ai/ja/job-execution/#job-execution-options

```bash
source /etc/profile.d/modules.sh
module load gcc/11.2.0
module load python/3.8/3.8.13
source /path/to/optenv/bin/activate
```
- ユーザプログラム実行に必要なモジュールの読み込みと仮想環境のactivateを行います．




<br>

## 3. 実行

作成した config.yaml と user.py が保存されているディレクトリに移動し，下記のコマンドで aiaccel を起動してください．

```bash
aiaccel-start --config config.yaml --clean
```
- コマンドラインオプション引数
    - `--config` - 設定ファイルを読み込むためのオプション引数です．読み込むコンフィグのパスを記述します．
    - `--clean` - aiaccel の起動ディレクトリ内に config.yaml の `workspace` で指定したディレクトリが存在する場合，削除してから実行するためのオプション引数です．



<br>

## 4. 最適化結果

- ハイパーパラメータ

    - x1
    - x2

- 評価値

    - polynomial

- 最適化手法

    - Sobol

- 結果比較

    - 最適化結果
        ```
        x1 = 3.134656548500061
        x2 = 1.9281481206417084

        polynomial = -6.967029595301102
        ```
