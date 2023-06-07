# Wrapper の作成例

必要に応じて wrapper プログラムを作成します．
aiaccel はユーザーが作成した最適化対象の関数の値を計算するプログラムの wrapper を作成するための API を提供します．

## 1. ファイル構成

### ユーザープログラム

- 与えられたパラメータから最適化したい目的関数の値を計算し，標準出力に出力します．

### wrapper.py

- aiaccel からパラメータをユーザープログラムに渡し，計算結果を aiaccel に返却します．

### config.yaml

- 最適化およびソフトウェアの設定ファイルです．


## 2. ファイル作成手順

### 関数プログラムの作成


以下のようなコマンドを実行した際に，最適化対象すべき値が標準出力に出力されるようなプログラムを作成します．

```console
> {cmd} --config={config} --trial_id={trial_id} --x1={x1} --x2={x2}
```
- **cmd** - ユーザープログラムを起動するコマンドです．
- **config** - コンフィグファイルのパスです．
- **trial_id** - aiaccel のジョブ ID です．
- **x1, x2, ...** - 最適化するパラメータです．ここでは例として 2 つのパラメータに x1, x2 という名前を付けましたが，任意の名前のパラメータを必要な数だけ設定することができます．

標準出力に出力される値は，以下のような形式である必要があります．

```
objective_y:{y}
```

- **y** - 最適化対象の計算結果です，


### ユーザープログラムの例

ここでは例として，python で最適化対象の関数を実装する場合を確認します．


```python
import argparse


def main(x1, x2):
    y = (x1 ** 2) - (4.0 * x1) + (x2 ** 2) - x2 - (x1 * x2)
    return y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--x1', type=float)
    parser.add_argument('--x2', type=float)
    args = vars(parser.parse_known_args()[0])

    y = main(args["x1"], args["x2"])

    print(f"objective_y:{y}")
```

#### モジュール
```python
import argparse
```
必要なモジュールをインポートします．

- argparse - コマンドライン引数を取得するために使用するモジュールです．

#### 最適化対象の関数
```python
def main(x1, x2):
    y = (x1 ** 2) - (4.0 * x1) + (x2 ** 2) - x2 - (x1 * x2)
    return y
```
最適化対象の関数を定義します．

#### 実行部分
```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--x1', type=float)
    parser.add_argument('--x2', type=float)
    args = vars(parser.parse_known_args()[0])

    y = main(args["x1"], args["x2"])

    print(f"objective_y:{y}")
```

以下の部分でパーサを作成し，コマンドライン引数を受け取ります．
```python
    parser = argparse.ArgumentParser()
    parser.add_argument('--x1', type=float)
    parser.add_argument('--x2', type=float)
    args = vars(parser.parse_known_args()[0])
```
この例にある最適化対象の関数では，コンフィグのパスとジョブの ID は使用しないため，パラメータ (`x1` と `x2`) のみを取得するような処理が行われています．


パラメータを最適化対象の関数 (`main()`) に渡し，値を計算します．
```python
    y = main(args["x1"], args["x2"])
```

計算結果を標準出力に出力します．
このとき，計算された値の前に "objective_y:" を付け加えます．
```python
    print(f"objective_y:{y}")
```

この python で実装されたプログラムの名前を user.py とすると，ユーザープログラムの起動コマンドは，`python user.py` となります．
例えばコンフィグのパスが `config.yaml`，ジョブの ID が 0，パラメータ `x1` が 1， パラメータ `x2` が 2 の場合，実行コマンドは次の通りです．
```console
> python user.py --config=condig.yaml --trial_id=0 --x1=1 --x2=2
```
このときの出力は以下のようになります．
```console
objective_y:-3.0
```

### wrapper.py の作成

以下のような wrapper プログラムを python で実装します．
```python
from aiaccel.util import aiaccel

run = aiaccel.Run()
run.execute_and_report("python user.py")
```

#### モジュール
```python
from aiaccel.util import aiaccel
```
- **aiaccel.util.aiaccel** - wrapper オブジェクトを作成するためのモジュールです．

#### Wrapper オブジェクトの作成
```python
run = aiaccel.Run()
```
aiaccel が提供する wrapper オブジェクトを作成します．

#### ユーザープログラムの実行
```python
run.execute_and_report("python user.py")
```
ユーザープログラムを実行します．
- `"python user.py"` の部分は，自身のプログラムを実行するためのコマンドを記述してください．
- コマンドライン引数として渡される config, trial_id, パラメータは， ***`run.execute_and_report()` の内部で自動的に追加されます***．そのため，ここに記述する必要はありません．


### config.yaml の作成

#### generic
```yaml
generic:
    workspace: "./work"
    job_command: "python wrapper.py"
    batch_job_timeout: 600
```

aiaccel で wrapper プログラムを最適化させる場合は，`job_command` に作成した wrapper の実行コマンドを設定します．
作成した python ファイルの名前が wrapper.py であれば，実行コマンドは `python wrapper.py` です．


#### resource
```yaml
resource:
  type: "local"
  num_workers: 4
```
wrapper プログラムを最適化する場合，指定可能な実行タイプは `"local"` または `"ABCI"` です．
`"python_local"` は選べません．
