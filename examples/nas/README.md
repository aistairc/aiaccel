# NAS(Network Architecture Search)の利用

NASは，ニューラルネットワークのネットワーク構造を自動的に探索し設計する手法です．
2024年4月現在，aiaccelのnasモジュールは，ProxylessNAS, Mnasnetをサポートしています．
本ガイドでは，ProxylessNASの例についてCIFAR10のデータセットを用いたネットワーク構造探索について説明します．

##  基本的な動作

本例は，基本的にaiaccelの１アプリケーションであり，config.yamlに記述されたハイパーパラメータ探索を行います．
そのハイパーパラメータ探索を行う実行対象がNASという構成になっています．

NAS自体がネットワーク構造探索をする機構であるため，本例はネットワーク構造探索をするNASのハイパーパラメータ最適化ということになります．
NASの計算量は非常に多く，更にそのハイパーパラメータ最適化となるとかなりの計算時間がかかることになるため，本例ではNASの構造探索やハイパーパラメータ最適化の設定をかなり省略し短時間で終了するよう設定されていますが，実際にNASやハイパーパラメータ最適化の機能を十分に利用したい場合は各種設定ファイルを見直してください．

## NASの動作

本例は，One-Shot NASの構造探索を取り扱います．
まずネットワーク構造を定義したファイルに記述されたスーパネットのトレーニングを行い，次にスーパネットからネットワーク構造を探索します．
設定次第では，これらのスーパネットのトレーニングとネットワーク構造の探索を同時に行うことも可能です．
ネットワーク構造が見つかったら，最後に再学習を行い発見したネットワーク構造の学習を終えモデルを生成します．


## examples/nasの構成

本exampleは，aiaccelの機能を用いて実行する例です．
aiaccelのインストールや実行の詳細については，トップディレクトリのREADME.mdを参照ください．
本exampleには，多くのファイルがありますが大まかに以下のように分類できます．

### ネットワーク構造を定義したファイル
- これらのファイルは，config_{探索空間}_{データセット}.yamlという命名規則のファイルであり，探索空間・データセットについての設定が記述されています．
今回は config_proxyless_cifar10.yaml を使用します．
以下は，本例で扱っている一覧です．
  - config_mnasnet_cifar10.yaml
  - config_mnasnet_imagenet.yaml
  - config_proxyless_cifar10.yaml
  - config_proxyless_imagenet.yaml
- 探索空間とデータセットは，examples/nas/nas_config.yamlに記述された設定から決定します．
探索空間については，nas_config.yaml内の nas.search_space から，データセットについては， dataset.name から読み出されます．
- 独自に定義した探索空間・データセットを利用したい場合は，[Tips](Tips)の[独自の探索空間・データセットを利用する方法](独自の探索空間・データセットを利用する方法)を参照してください．

### aiaccelのファイル群
- aiaccel関連のファイルは以下の通りです．
用途はaiaccelの通常の機能と変わらないので，詳細についてはトップディレクトリのREADME.mdを参照ください．
  - config.yaml
  - job_script_preamble.sh
  - user.py

### NAS関連のファイル
- nas_config.yaml には，NASに関する設定は，全てこのファイルに記述されています．
- dataset
  - name: データセット名を指定します．ここで指定したデータセット名は，[ネットワーク構造を定義したファイル](ネットワーク構造を定義したファイル)にて説明されたように参照されるネットワーク構造とデータセットの定義ファイル名に利用されますので，編集する場合は[独自の探索空間・データセットを利用する方法](独自の探索空間・データセットを利用する方法)を理解した上で行って下さい．
  - num_data_archtecture_search: ネットワーク構造探索に利用するデータセットのデータ数を指定する．
  - num_search_classes: 指定したデータセットのクラス数を指定する．
  - path: データセットを保存するパスを指定する．１度データセットをダウンロードすれば，以降は再利用することでダウンロード時間は必要無い．
  - retrain: 再学習に利用するデータセットのパスを指定する．
- environment
  - device_ids: 複数GPUを利用する場合，インデックスをリストで指定する．
  - gpus: 複数GPUを利用する場合，その数を指定する．
  - num_workers: データローダーのワーカー数を指定する．
  - result_dir: ネットワーク構造探索の結果を保存するディレクトリを指定する．
- nas
  - num_epochs_architecture_search: NASの構造探索を行うエポック数を指定する．
  - num_epochs_supernet_train: NASのスーパネット学習を行うエポック数を指定する．
  - num_epochs_retrain: NASの再学習を行うエポック数を指定する．
  - search_space: 探索空間を指定する．ここで指定した探索空間名は，[ネットワーク構造を定義したファイル](ネットワーク構造を定義したファイル)にて説明されたように参照されるネットワーク構造とデータセットの定義ファイル名に利用されますので，編集する場合は[独自の探索空間・データセットを利用する方法](独自の探索空間・データセットを利用する方法)を理解した上で行って下さい．
  - search_space_config_path: [ネットワーク構造を定義したファイル](ネットワーク構造を定義したファイル)があるディレクトリを指定する．
  - seed: NASにおけるシード値を指定する．
  - skip_architecture_search: 構造探索をスキップするかを指定する．
  - skip_retrain: 再学習をスキップするかを指定する．
  - skip_train: スーパネット学習をスキップするかを指定する．
  - type: スーパネット学習と構造探索を同時実行するか分離実行するかを指定する．

- retrain:
  - hyperparameters:
    - base_lr: 再学習のハイパーパラメータである学習率を指定する．
    - batch_size: 再学習のハイパーパラメータであるバッチサイズを指定する．
    - momentum: 再学習のハイパーパラメータであるモーメンタムを指定する．
  - num_epochs: 再学習のエポック数を指定する．
  - weight_decay: 重み減衰を指定する．

- trainer:
  - accelerator: aiaccel.nasモジュールで使用するLightning.Trainerのアクセラレータを指定する．
  - enable_model_summary: Lightning.Trainerがモデルのサマリーを表示するか指定する．
  - enable_progress_bar: Lightning.Trainerがプログレスバーを表示するか指定する．
  - logger: Lightning.Trainerのロガーを指定する．
  - strategy: Lightning.Trainerのトレーニング・評価・予測にわたるモデル分布制御の戦略名を指定します．具体的には，ddp, ddp_spawn, ipuなどが指定できます．詳細は Lightningのリファレンスを参照してください．
- trainer_config.yaml
  - aiaccel.nasモジュールで利用されるLightning.TrainerのOmegaconfによるinstantiateをuser.pyにて補助します．nas_config.yamlのファイル名を変更したい場合などに編集してください．

## examples/nasの実行
- ここでは，実際の実行方法について説明します．

### ABCI上での実行方法
- 基本的には，aiaccelの実行方法を変わらず以下のコマンドにより実行します．
```shell
$ aiaccel-start --config config.yaml
```
- 本例では，細かい設定は nas_config.yaml に記述されています．注意点としては，４GPUを搭載した計算ノード上で実行する設定となっているため，計算ノードのリソースを変更したい場合は，nas_config.yaml の説明を読み適宜変更してください．
- 計算ノードを変更する具体的な編集内容は，以下の通りです．
  - nas_config.yamlのenvironment.device_idsやenvironment.gpusがGPU数の変更に関連し，environment.num_workersはCPU数に関連します．
  - trainer.strategyは，DDP(Distributed Data Parallel)を利用したい場合などは変更して下さい．


### Mac OS上での実行方法
- Mac OSで実行するための nas_config_macos.yaml が用意されていますので，適宜コピーして利用してください．
- 基本的には，aiaccelの実行方法を変わらず以下のコマンドにより実行します．
```shell
$ aiaccel-start --config config.yaml
```
- 本例では，ABCI上で実行する例として設定されているため，いくつかの設定項目を編集する必要があります．
- config.yaml
  - resource.typeを python_local に変更し，num_node は 1 としてください．
  - optimize.trial_number は大きいと計算時間が非常に長くなるため，小さい数値に変更することをおすすめします．
- nas_config.yaml
  - environment.device_ids は null に，num_workers は CPU数に基づき設定，gpusは 1 などに設定してください．
  - trainer.acceleratorを mps に，strategyを auto に設定してください．

# Tips
## 独自の探索空間・データセットを利用する方法
- WIP


