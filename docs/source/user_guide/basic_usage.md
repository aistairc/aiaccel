# 基本的な使い方

## ABCIのセットアップ
ABCIのセットアップは下記資料を参考ください。

[https://docs.abci.ai/ja/](https://docs.abci.ai/ja/)


## Python-venvによる仮想環境の作成

venv環境での使用を推奨いたします。このチュートリアルはvenv環境で動作させることを前提としています。

~~~bash
python -m venv optenv
~~~

ここでは仮想環境の名前を「optenv」とし、以後も当仮想環境を「optenv」と表記します。
仮想環境の名前は任意の名前を設定できます。

## アクティベート
仮想環境を利用するには下記コマンドを実施します。
~~~bash
source optenv/bin/activate
~~~
以後の作業はアクティベート済みのものとして進めます

## インストール
aiaccelをダウンロードします。

~~~bash
git clone https://github.com/aistairc/aiaccel.git
~~~
ダウンロード完了後、aiaccelフォルダに移動します。

~~~bash
cd aiaccel
~~~

依存環境をインストールします.

~~~bash
python -m pip install -r requirements.txt
~~~

```{note}
事前にpipをアップグレードすることを推奨いたします。
```

~~~bash
python -m pip install --upgrade pip
~~~

setup.pyを実行し、aiaccelをインストールします。

~~~bash
python setup.py install
~~~

または、

~~~bash
python -m pip install git+https://github.com/aistairc/aiaccel.git
~~~

aiaccelがインポートできることを確認します。

~~~bash
python
import aiaccel
~~~

## チュートリアル


1. プロジェクトファイルの構成

コンフィグファイル、ユーザープログラム、ABCI実行用シェルスクリプトを用意し、一つのフォルダに格納します。
以後、この一式を含んだフォルダを「プロジェクトフォルダ」とします。
プロジェクトフォルダは任意の場所に作成してください。

```
├── config.yaml
├── user.py
└── job_script_preamble.sh
```

```{note}
- `config.yaml` - 最適化の設定ファイルです。
- `user.py` - 最適化対象のプログラムです。詳細は後述します。
- `job_script_preamble.sh` - ABCIにジョブを投入するためのスクリプトです。このファイルにはスクリプトの共通部分のみを記述します。このファイルをベースに、バッチジョブファイルを生成します。
```

```{note}
`config.yaml`、 `user.py`、 `job_script_preamble.sh` は任意のファイル名に変更可能です。
```

2. コンフィグファイルの作成

### generic

**サンプル**

```yaml
generic:
    workspace: "./work"
    job_command: "python user.py"
    batch_job_timeout: 600
```

- **workspace** - 途中経過の保存先を指定します。
- **job_command** - ユーザプログラムを実行するコマンドを記述します。
- **batch_job_timeout** - jobのタイムアウト時間を設定します。[単位: 秒]

### resource

**サンプル**

```yaml
resource:
    type: "ABCI"
    num_node: 4
```

- **type** - 実行環境を指定します。 `ABCI` 、または `local` を指定します。
- **num_node** - 使用するノード数を指定します。ローカルの場合はCPUコア数を指定してください。

### ABCI

**サンプル**

```yaml
ABCI:
    group: "[group]"
    job_script_preamble: "./job_script_preamble.sh"
    job_execution_options: ""
```

- **job_script_preamble** - ABCI上でソフトウェアを実行するためのラッパーシェルスクリプトです。詳細は後述します。
- **group** - 自分が所属しているABCIグループ名を指定します。([]は記述不要です。)

### optimize

**サンプル**

```yaml
optimize:
search_algorithm: "nelder-mead"
goal: "minimize"
trial_number: 30
rand_seed: 42
parameters:
    -
    name: "x1"
    type: "uniform_float"
    lower: 0.0
    upper: 5.0
    initial: 1.0
    -
    name: "x2"
    type: "uniform_float"
    lower: 0.0
    upper: 5.0
    initial: 1.0
```

- **search_algorithm** - 最適化アルゴリズムを指定します。
- **goal** - 最適化の方向を設定します。[minimize | maximize]
- **trial_number** - 試行回数を設定します。
- **parameters**
- **name** - ハイパーパラメータの名前を設定します。
- **type** - ハイパーパタメータのデータ型を設定します。
    - データ型一覧
        - uniform_float
        - uniform_int
        - categorical
        - ordinal
        - sequential
- **lower** - ハイパーパラメータ最小値を設定します。
- **upper** - ハイパーパラメータ最大値を設定します。
- **initial** - ハイパーパラメータの初期値を設定します。
- **step**  - ハイパーパラメータの分解能を設定します(最適化アルゴリズムがgridの場合は必ず指定してください。)。
- **log** - 対数設定用の項目です(最適化アルゴリズムがgridの場合は必ず指定してください。)。
- **base** - 対数設定用の項目です(最適化アルゴリズムがgridの場合は必ず指定してください。)。
- **comment** - 自由記述欄。


```{note}
aiaccelは、次の最適化アルゴリズムをサポートしています。
- **random** - ハイパーパラメータの値をランダムに生成します。
- **grid** - ハイパーパラメータの値を一定間隔でサンプリングします。
- **sobol** - Sobol列を用いてハイパーパラメータの値を生成します。
- **nelder-mead** - ヒューリスティクスな最適化アルゴリズムです.
- **tpe** - ベイズ最適化による最適化アルゴリズムです。
```

### parametersの記述例

#### Type: uniform_intの記述例

```yaml
parameters:
    -
        name: "x1"
        type: "uniform_int"
        lower: 0
        upper: 5
        initial: 1
    -
        name: "x2"
        type: "uniform_int"
        lower: 0
        upper: 5
        initial: 1
```

```{note}
- initialを指定しない場合は、項目を削除します。
```
```yaml
-
    name: "x1"
    type: "uniform_int"
    lower: 0
    upper: 5
```

#### Type: uniform_floatの記述例

```yaml
parameters:
    -
        name: "x1"
        type: "uniform_float"
        lower: 0.0
        upper: 5.0
        initial: 0.0
    -
        name: "x2"
        type: "uniform_float"
        lower: 0.0
        upper: 5.0
        initial: 0.0
```

#### Type: categoricalの記述例

```yaml
parameters:
    -
        name: "x1"
        type: "categorical"
        choices: ['green', 'red', 'yellow', 'blue']
    -
        name: "x2"
        type: "categorical"
        choices: ['green', 'red', 'yellow', 'blue']
```

```{note}
- categorial使用時は `choices` 項目を使用します. `choices` は配列で指定する必要があります。
- catogoricalを使用できるのは、最適化アルゴリズムが `Random` と `TPE` の場合のみです。
```

#### Type: ordinalの記述例

```yaml
parameters:
    -
        name: "x1"
        type: "ordinal"
        sequence: [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        lower: 0
        upper: 1024
    -
        name: "x2"
        type: "ordinal"
        sequence: [1024, 512, 256, 128, 64, 32, 16, 8, 4, 2]
        lower: 0
        upper: 1024
```

```{note}
- ordinal使用時は `sequence` 項目を使用します. `sequence` は配列で指定する必要があります。
- ordinal使用時は `initial` の設定はできません。
- ordinalを使用できるのは、最適化アルゴリズムが `RandomSearch` と `TPE` の場合のみです。
```

### grid使用時の注意事項
最適化アルゴリズムで `grid` を使用する場合、 `parameters` の設定に `log` 、 `step` 、 `base` を指定してください。

```yaml
parameters:
    -
        name: "x1"
        type: "uniform_int"
        lower: 0
        upper: 5
        step: 1
        log: false
        base: 10
        initial: 0.0
    -
        name: "x2"
        type: "uniform_int"
        lower: 0
        upper: 5
        step: 1
        log: false
        base: 10
        initial: 0.0
```


### Nelder-Mead使用時の注意事項
Nelder-Meadを使用する場合、 `initial` を配列で指定する必要があります。

```yaml
parameters:
    -
        name: "x1"
        type: "uniform_int"
        lower: 0
        upper: 5
        initial: [0, 5, 3]
    -
        name: "x2"
        type: "uniform_int"
        lower: 0
        upper: 5
        initial: [2, 4, 1]
```

また、 `initial` を使用しない場合は、空のリストを指定します.

```yaml
parameters:
    -
        name: "x1"
        type: "uniform_int"
        log: False
        lower: 0
        upper: 5
        initial: []
    -
        name: "x2"
        type: "uniform_int"
        log: False
        lower: 0
        upper: 5
        initial: []
```

あるいは、 `initial` 項目そのものを削除します。

```yaml
parameters:
    -
        name": "x1"
        type": "uniform_int"
        log": False
        lower": 0
        upper": 5
    -
        name: "x2"
        type: "uniform_int"
        log: False
        lower: 0
        upper: 5
```

### コンフィグファイル サンプル

config.yaml

```yaml
generic:
    workspace: "./work"
    job_command: "python user.py"
    batch_job_timeout: 600

resource:
    type: "local"
    num_node: 4

ABCI:
    group: "[group]"
    job_script_preamble: "./job_script_preamble.sh"
    job_execution_options: ""

optimize:
    search_algorithm: "nelder-mead"
    goal: "minimize"
    trial_number: 30
    rand_seed: 42
    parameters:
        -
            name: "x1"
            type: "uniform_float"
            lower: 0.0
            upper: 5.0
            initial: 1.0
        -
            name: "x2"
            type: "uniform_float"
            lower: 0.0
            upper: 5.0
            initial: 1.0
```

3. ユーザープログラムの作成
最適化対象の処理を作成します。ここでは、作成済みモデルをaiaccelで最適化するための変更方法を記述します。

次の関数を最適化させる場合の例を示します。

```python

    def func(x1, x2):
        y = (x1**2) - (4.0 * x1) + (x2**2) - x2 - (x1 * x2)
        return y
```

これを、aiaccelで最適化させるには次のように変更します。

```python
    from aiaccel.util import opt

    def func(p):
        x1 = p["x1"]
        x2 = p["x2"]
        y = (x1**2) - (4.0 * x1) + (x2**2) - x2 - (x1 * x2)
        return y

    if __name__ == "__main__":

        run = opt.Run()
        run.execute_and_report(func)
```

4. Wrapperの作成
必要に応じてwrapperプログラムを作成します。aiaccelはユーザープログラムのwrapperを作成するためのAPIを提供します。

**サンプル**

wrapper.py(任意の名前に変更可能)
```python
    from aiaccel.util import aiaccel

    # Wrapperオブジェクトの生成
    run = aiaccel.Run()

    # ユーザープログラムを実行します。
    # commandにユーザープログラムを実行するためのコマンドを記述してください。
    # コマンドライン引数は自動で生成します。
    #  --config
    #  --index
    #  --x1 (例)
    #  --・・・
    run.execute_and_report("python user.py")
```

aiaccelでwrapperプログラムを最適化させる場合はコンフィグファイルの`job_command`を変更します。

```python
    generic:
        workspace: "./work"
        job_command: "python wrapper.py"
        batch_job_timeout: 600
```

5. job_script_preamble.shの作成
`job_script_preamble.sh` は、ABCIにジョブを投入するためのバッチファイルのベースファイルです。
このファイルには事前設定を記述します。ここに記述した設定が全てのジョブに適用されます。

**サンプル**

~~~bash
    #!/bin/bash

    #$-l rt_C.small=1
    #$-j y
    #$-cwd

    source /etc/profile.d/modules.sh
    module load gcc/11.2.0
    module load python/3.8/3.8.13
    module load cuda/10.2
    module load cudnn/8.0/8.0.5
    module load nccl/2.8/2.8.4-1
    source ~/optenv/bin/activate

    AIACCELPATH=$HOME/local/aiaccel-dev
    export PYTHONPATH=$AIACCELPATH:$AIACCELPATH/lib
~~~

6. 最適化実行
プロジェクトフォルダに移動し、次のコマンドを実行します。
~~~bash
python -m aiaccel.start --config config.yaml
~~~

```{note}
    コンフィグファイル名 `config.yaml` は適切な文字列に変更してください。
```

実行するとターミナルに進捗状況を出力します。

### オプション付きの実行

`start` コマンドの後に、追加オプションを指定できます。

~~~bash
python -m aiaccel.start
~~~

- --clean : workspaceが既に存在する場合、最適化実行前にworkspaceを削除します。
- --resume : workspaceが既に存在する場合、保存データが存在するトライアルを指定することで、指定のトライアルから再開することができます。

### 例
~~~bash
python -m aiaccel.start --config config.yaml --clean
~~~

### ローカル環境での実行方法
ローカル環境でaiaccelを使用する場合は、次のように設定を変更します。

#### resourceの設定
コンフィグファイルの `resource` の `type` に `local` を指定します。

```yaml
resource:
    type: "local"
    num_node: 4
```

#### ABCIの設定
ローカル環境で実施する場合, `ABCI` の設定は動作に反映されません。

```yaml
ABCI:
    group: "[group]"
    job_script_preamble: "./job_script_preamble.sh"
    job_execution_options: ""
```

#### job_script_preamble.sh
ローカル環境で実施する場合、 `job_script_preamble.sh` は不要です。
記述した内容は動作に反映されません。
