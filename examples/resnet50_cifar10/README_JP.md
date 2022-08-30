# README_JP

## 1. ファイル構成

### config.yaml

- 最適化およびソフトウェアの設定ファイルです。

### job_script_preamble.sh

- ABCIにジョブを投入するためのバッチファイルのベースファイルです。

### setup_dataset.py

- データセットcifar10のダウンロード用プログラムです。

### user.py

- pytorchを用いて、モデルresnet50にデータセットcifar10を学習させるユーザプログラムです。

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

- **batch_job_timeout** -  jobのタイムアウト時間を設定します。[単位: 秒]
    - ご参考 - 100epochの学習に最長60分程度掛かるので、長めに`7200`で設定します。

#### resource
```yaml
resource:
  type: "abci"
  num_node: 6
```

- **type** - 実行環境を指定します。学習にGPUを用いるため、`"abci"`で設定します。
- **num_node** - 使用するノード数を指定します。
    - ご参考 - 最適化手法は`nelder-mead`、パラメータ数は`5`なので、同時に計算されるシンプレックス頂点の最大数である`6`で設定します。

#### ABCI
```yaml
ABCI:
  group: "[group]"
  job_script_preamble: "./job_script_preamble.sh"
  job_execution_options: ""

```

- **group** - 自分が所属しているABCIグループ名を指定します。

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

- **search_algorithm** - 最適化アルゴリズムを指定します。
- **goal** - 最適化の方向を設定します。
    - ご参考 - Validation Error Rateを最小化することが目的であるため、`"minimize"`で設定します。
- **trial_number** - 試行回数を設定します。
- **parameters** - ハイパーパラメータの各種項目を設定します。
    - **name** - ハイパーパラメータの名前を設定します。
    - **type** - ハイパーパタメータのデータ型を設定します。
    - **lower / upper** - ハイパーパラメータ最小値 / 最大値を設定します。
    - **initial** - ハイパーパラメータの初期値を設定します。

### user.py の作成
---

#### train_func

- `main`内で用いている訓練用関数です。

#### val_test_func

- `main`内で用いている評価・汎化性能検証用関数です。

#### main

- 最適化対象のメイン関数です。
    - この関数の`return`値を最適化します。`Validation Error Rate`で設定しています。

### job_script_preamble.sh の作成
---

```
#!/bin/bash

#$-l rt_F=1
#$-j y
#$-cwd
#$ -l h_rt=2:00:00
```
- ABCIのバッジジョブ実行オプションを指定しています。`#$-l rt_F=1`でFullノードを利用するように設定しています。
    - 参考: https://docs.abci.ai/ja/job-execution/#job-execution-options

```
source /etc/profile.d/modules.sh
module load gcc/11.2.0 python/3.8/3.8.13 cuda/10.1/10.1.243 cudnn/7.6/7.6.5
source ../../../../resnet50sample_env/bin/activate
```
- ユーザプログラム実行に必要なモジュール読み込み・仮想環境のactivateを行います。

<br>

## 3. 動作説明
- optとpytorchが動作する環境が必要です。

    - ABCIにおけるpytorch導入手順(出典:https://docs.abci.ai/ja/apps/pytorch/)
        - optの仮想環境作成・activate後、下記コマンドを実行してください。

            ```bash
            pip3 install --upgrade pip setuptools
            pip3 install filelock torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
            ```

- config.yaml の [ABCI][group] は、自分が所属しているABCIグループ名に変更してください。

    ```yaml
    ABCI:
        group: "[group]"
        job_script_preamble: "./job_script_preamble.sh"
        job_execution_options: ""
    ```

- job_script_preamble.sh の9行目は、適切な仮想環境のactivate文に書き換えてください。

    ```bash
    source ../../../../resnet50sample_env/bin/activate
    ```

- 事前に setup_dataset.py を実行し、データセットのダウンロードを行ってください。

    ```bash
    python setup_dataset.py
    ```

- 上記準備が完了したら、下記コマンドでoptを起動してください。

    ```bash
    python -m aiaccel.start --config config.yaml --clean
    ```

<br>

## 4. 最適化結果

- ハイパーパラメータ

    - batch_size
    - lr
    - momentum
    - weight_decay
    - lr_decay

- 評価値

    - Validation Error Rate

- 最適化手法

    - Nelder-Mead

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
- 上記設定で最適化を実行すると、ABCIポイントを約50ポイント消費します。

