# フラクタルデータセットの転移学習最適化の ABCI 環境での実行例

ここでは，フラクタルデータセットの転移学習最適化を ABCI 環境で実行する方法を説明します．例として，モデル ResNET50 にフラクタルデータセットを事前学習、 データセット CIFAR10 を転移学習させる際のハイパパラメータの最適化を行います．

なお、データセットのコンテスト用exampleのため、デフォルトでは 1 trialのみの実行になっています.

以下の説明では aiaccel/examples/resnet50_flactal_cifar10 に保存されているファイルを編集して使用します．



## 1. ファイル構成

### config.yaml

- 最適化およびソフトウェアの設定ファイルです．


### job_script_preamble.sh

- ABCI で使用するモジュール指定やジョブ設定を行うためのシェルスクリプトファイルです．


### FractalDB-Pretrained-ResNet-PyTorch

- フラクタルデータセットの生成・事前学習・CIFAR10への転移学習を行うプログラムを含んだディレクトリです.
- 詳細は下記 git も参考にしてください.
https://github.com/hirokatsukataoka16/FractalDB-Pretrained-ResNet-PyTorch


### requirements.txt

- FractalDB-Pretrained-ResNet-PyTorch の動作に必要なモジュールをインストールするためのファイルです.
- aiaccel 自体のインストールは個別に行う必要があります.


<br>

## 2. ファイル作成手順

### config.yaml の作成

#### generic
```yaml
generic:
  workspace: "./work"
  job_command: "bash exe_parallel.sh"
  batch_job_timeout: 180000
```
- **workspace** - aiaccel の実行に必要な一時ファイルを保存するディレクトリを指定します．
- **job_command** - ユーザープログラムを実行するためのコマンドです．
- **batch_job_timeout** - ジョブのタイムアウト時間を設定します．[単位: 秒]
    - 参考 - フラクタルデータセットの生成・事前学習・CIFAR10への転移学習に最長 48 時間程かかるため，`180000` と長めに設定します．

#### resource
```yaml
resource:
  type: "abci"
  num_workers: 4
```

- **type** - 実行環境を指定します．ABCI 環境で実行するためには `"abci"` で設定します．
- **num_workers** - 使用するノード数を指定します．


#### ABCI
```yaml
ABCI:
  group: "[group]"
  job_script_preamble: "./job_script_preamble.sh"
  job_execution_options: ""

```

- **group** - 所属している ABCI グループを指定します．
- **job_script_preamble** - ABCI の設定を記述したシェルスクリプトのファイルを指定します．


#### optimize
```yaml
optimize:
  search_algorithm: "aiaccel.optimizer.RandomOptimizer"
  goal: "minimize"
  trial_number: 1
  rand_seed: 42
  parameters:
    -
      name: "dummy parameter"
      type: "categorical"
      choices: ["dummy"]

```

- **search_algorithm** - 最適化アルゴリズムを設定します．この例ではランダムオプティマイザを設定しています．
- **goal** - 最適化の方向を設定します．
- **trial_number** - 試行回数を設定します．今回は 1 trialのみとしています.
- **rand_seed** - 乱数の生成に使用するシードを設定します．
- **parameters** - ハイパパラメータの各種項目を設定します．今回は実際はユーザプログラム内では使用しない、ダミーのハイパパラメータを設定しています.


### job_script_preamble.shの作成

`job_script_preamble.sh` は、ABCI にジョブを投入するためのバッチファイルのベースファイルです．
このファイルには事前設定を記述します．
ここに記述した設定が全てのジョブに適用されます．

```bash
#!/bin/bash

#$-l rt_F=1
#$-j y
#$-cwd
#$ -l h_rt=50:00:00
```

- ABCIのバッチジョブ実行オプションを指定しています．`#$-l rt_F=1`でFullノードを利用するように設定しています．
    - 参考: https://docs.abci.ai/ja/job-execution/#job-execution-options

```bash
source /etc/profile.d/modules.sh
module load gcc/12.2.0 python/3.10/3.10.10 cuda/12.2/12.2.0 cudnn/8.9/8.9.2
source ./work/bin/activate
cd FractalDB-Pretrained-ResNet-PyTorch
```
- ユーザプログラム実行に必要なモジュールの読み込みと仮想環境の activate を行います．
- ./work/bin/activate には aiaccel をインストールした仮想環境のパスを設定します．


<br>


## 3. 動作説明
- aiaccel の動作する環境作成・activate後に、下記コマンドを実行して FractalDB-Pretrained-ResNet-PyTorch 用のモジュールをインストールしてください.

    ```bash
    cd examples/resnet50_fractal_cifar10/
    pip install -r requirements.txt 
    ```

- config.yaml の [ABCI][group] は，所属しているABCIグループ名に変更してください．

    ```yaml
    ABCI:
        group: "[group]"
        job_script_preamble: "./job_script_preamble.sh"
        job_execution_options: ""
    ```

- job_script_preamble.sh の仮想環境パスを自身の仮想環境のものに変更してください.

    ```bash
    source ./work/bin/activate
    ```

- 下記 git を参考に、FractalDB-Pretrained-ResNet-PyTorch/data ディレクトリ直下に CIFAR10 のデータセットを配置してください.
  https://github.com/chatflip/ImageRecognitionDataset


- CIFAR10/test ディレクトリ名を CIFAR10/val に変更してください.

    ```bash
    mv FractalDB-Pretrained-ResNet-PyTorch/data/CIFAR10/test FractalDB-Pretrained-ResNet-PyTorch/data/CIFAR10/val
    ```

- 上記準備を終えたら，下記のコマンドで aiaccel を起動してください．

    ```bash
    aiaccel-start --config config.yaml --clean
    ```
- コマンドラインオプション引数
    - `--config` - 設定ファイルを読み込むためのオプション引数です．読み込むコンフィグのパスを記述します．
    - `--clean` - aiaccel の起動ディレクトリ内に config.yaml の workspace で指定したディレクトリが存在する場合，削除してから実行するためのオプション引数です．

<br>

## 4. 結果の確認

CIFAR10 の転移学習時の validation loss が、aiaccel側に出力されます.
デフォルト状態での実行結果は、0.341864 になります.