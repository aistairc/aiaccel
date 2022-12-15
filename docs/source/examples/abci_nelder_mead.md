# ネルダーミードオプティマイザの ABCI 環境での実行例

ここでは，ネルダーミードオプティマイザをABCI環境で実行する方法を説明します．例として，モデル ResNET50 に データセット CIFAR10 を学習させる際のハイパーパラメータの最適化を行います．

以下の説明では aiaccel/examples/resnet50_cifar10 に保存されているファイルを編集して使用します．



## 1. ファイル構成

### config.yaml

- 最適化およびソフトウェアの設定ファイルです．


### user.py

- 与えられたパラメータから目的関数の値を計算し，aiaccel の Storage に保存するユーザプログラムです．今回の例では，モデル ResNET50 にデータセット CIFAR10 を学習させるユーザプログラムです．


### job_script_preamble.sh

- ABCIで使用するモジュール指定やジョブ設定を行うためのシェルスクリプトファイルです．


### setup_dataset.py

- データセット CIFAR10 ダウンロード用プログラムです．


<br>

## 2. ファイル作成手順

### config.yaml の作成
---

#### generic
```yaml
generic:
  workspace: "./work"
  job_command: "python user.py"
  batch_job_timeout: 7200
```
- **workspace** - aiaccel の実行に必要な一時ファイルを保存するディレクトリを指定します．
- **job_command** - ユーザープログラムを実行するためのコマンドです．
- **batch_job_timeout** - job のタイムアウト時間を設定します．[単位: 秒]
    - 参考 - 100 epoch の学習に最長 60 分程かかるため，`7200` と長めに設定します．

#### resource
```yaml
resource:
  type: "abci"
  num_node: 6
```

- **type** - 実行環境を指定します。ABCI環境で実行するためには `"abci"` で設定します．
- **num_node** - 使用するノード数を指定します．
    - 参考 - 今回の例では，最適化アルゴリズムが `NelderMeadOptimizer`，パラメータ数が`5`のため，
    同時に計算されるシンプレックス頂点の最大数である`6`にノード数を設定します．


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
  search_algorithm: 'aiaccel.optimizer.NelderMeadOptimizer'
  goal: "minimize"
  trial_number: 100
  rand_seed: 42
  parameters:
    -
      name: "batch_size"
      type: "uniform_int"
      lower: 64
      upper: 256
      initial: 256
    -
      name: "lr"
      type: "uniform_float"
      lower: 1.0e-4
      upper: 1.0
      initial: 0.1
    -
      name: "momentum"
      type: "uniform_float"
      lower: 0.8
      upper: 1.0
      initial: 0.9
    -
      name: "weight_decay"
      type: "uniform_float"
      lower: 5.0e-6
      upper: 5.0e-2
      initial: 5.0e-4
    -
      name: "lr_decay"
      type: "uniform_float"
      lower: 0.0
      upper: 1.0
      initial: 1.0e-3

```

- **search_algorithm** - 最適化アルゴリズムを設定します．この例ではネルダーミードオプティマイザを設定しています．
- **goal** - 最適化の方向を設定します．
    - 参考 - Validation Error Rate を最小化することが目的であるため，`"minimize"` と設定します．
- **trial_number** - 試行回数を設定します．
- **rand_seed** - 乱数の生成に使用するシードを設定します．
- **parameters** - ハイパパラメータの各種項目を設定します．ここでは，5 種類のハイパパラメータを用意しています．5 つのパラメータに対して，以下の項目をそれぞれ設定する必要があります．
    - **name** - ハイパパラメータの名前を設定します．
    - **type** - ハイパパラメータのデータ型を設定します．
    - **lower / upper** - ハイパパラメータ最小値 / 最大値を設定します．
    - **initial** - ハイパパラメータの初期値を設定します．`NelderMeadOptimizer`の場合は，基本的にシンプレックスの頂点の数と同じ要素数をもつリストを設定します．しかし，仮に頂点の数未満の数値リストを与えた場合でも，aiaccel 内部の処理により足りない個数分の数値が適当に設定されます．今回の例では，各ハイパパラメータに初期値として１個の数値しか与えていませんが，足りない５個の数値については内部処理で適当に設定されています．



### user.py の作成
---

`user.py` は以下のような構成になっています．

#### train_func

- `main` 内で用いている訓練用関数です．

#### val_test_func

- `main` 内で用いている評価・汎化性能検証用関数です．

#### main

- 最適化対象のメイン関数です．この関数の `return` 値を最適化します．`Validation Error Rate` で設定しています．


### job_script_preamble.shの作成
---

`job_script_preamble.sh` は、ABCI にジョブを投入するためのバッチファイルのベースファイルです．
このファイルには事前設定を記述します．
ここに記述した設定が全てのジョブに適用されます．

```bash
#!/bin/bash

#$-l rt_F=1
#$-j y
#$-cwd
#$ -l h_rt=2:00:00
```

- ABCIのバッチジョブ実行オプションを指定しています．`#$-l rt_F=1`でFullノードを利用するように設定しています．
    - 参考: https://docs.abci.ai/ja/job-execution/#job-execution-options

```bash
source /etc/profile.d/modules.sh
module load gcc/11.2.0 python/3.8/3.8.13 cuda/10.1/10.1.243 cudnn/7.6/7.6.5
source /path/to/optenv/bin/activate
```
- ユーザプログラム実行に必要なモジュールの読み込みと仮想環境のactivateを行います．


<br>


## 3. 動作説明
- aiaccel と PyTorch 動作する環境が必要です．
    - ABCIにおけるPyTorch導入手順(出典:https://docs.abci.ai/ja/apps/pytorch/)
        - aiaccelの仮想環境作成・activate後、下記コマンドを実行してください．

            ```bash
            pip3 install --upgrade pip setuptools
            pip3 install filelock torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
            ```

- config.yaml の [ABCI][group] は，所属しているABCIグループ名に変更してください．

    ```yaml
    ABCI:
        group: "[group]"
        job_script_preamble: "./job_script_preamble.sh"
        job_execution_options: ""
    ```

- 事前に `python3 setup_dataset.py` を実行し，データセットのダウンロードを行ってください．

- 上記準備を終えたら，下記のコマンドで aiaccel を起動してください．

```bash
aiaccel-start --config config.yaml --clean
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

    - batch_size
    - lr
    - momentum
    - weight_decay
    - lr_decay

- 評価値

    - Validation Error Rate

- 最適化アルゴリズム

    - NelderMeadOptimizer

- 結果比較

    - デフォルトパラメータ
        ```
        batch_size = 256,
        lr = 0.1,
        momentum = 0.9,
        weight_decay = 5.0e-4,
        lr_decay = 1.0e-3

        EvalLoss = 0.9949815124511718, evalAcc = 79.86

        TestLoss = 0.9659099947929383, TestAcc = 80.95
        ```

    - 最適化結果
        ```
        batch_size = 226,
        lr = 0.036964256184365454,
        momentum = 0.8899804003224022,
        weight_decay = 0.0027056323588476026,
        lr_decay = 0.03683728125425209

        EvalLoss = 0.7713460917830467, evalAcc = 83.27

        TestLoss = 0.7550274838387966, TestAcc = 83.59
        ```

<br>

## 5. ご注意
- 上記設定で最適化を実行すると，ABCIポイントを約50ポイント消費します.