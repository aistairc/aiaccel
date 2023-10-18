# aiaccelの概要
aiaccelは，与えられた入力ハイパーパラメータ群から最適なハイパーパラメータを出力するハイパーパラメータ最適化ライブラリです．
aiaccelは，ABCIのインタラクティブノード上で実行することを想定しています。
ABCIについては，[ABCI User Guide](https://docs.abci.ai/ja/)を参照ください．

また、ローカル環境でも利用可能です。

aiaccelは、内部状態や最適化結果をデータベースで管理します。最適化対象のプログラム(ここでは `User Program`と呼称)の実行タスクをABCI計算ノードに渡し、結果をデータベースに保存します。

![aiaccel_system](images/aiaccel_system.png)

## aiaccelの入出力

aiaccelの入出力について解説します．

![aiaccel_input_output](images/aiaccel_input_output.png)

- 入力
  - `Config` - コンフィグレーションファイルです．最適化のパラメータの設定、最適化アルゴリズムの設定、最適化対象(User Program)等を記述します。コンフィグレーションファイルは、Yaml または JSON形式で記述します.
  <br>
    - 例：examples/sphere/config.yaml を参照ください．

  - `User Program` - 最適化対象のプログラムです．user.py はコンフィグレーションファイルで指定します．<br>
    - 例： examples/sphere/user.py を参照ください．

  - `Job Script` - ジョブスクリプトファイルです．
  ジョブスクリプトファイルは，ABCIで実行するジョブを記述します．
  aiaccelは、指定したジョブスクリプトファイルを元に、新たにジョブスクリプトファイルを生成します．ここで指定するジョブスクリプトは、load moduleなどの事前処理を記述します．
    - 例： examples/sphere/job_script_preamble.sh を参照ください．


- 出力
  - `Work Directory` - aiaccel実行時に生成されるワークディレクトリです(以下、workと記述します)．workはコンフィグレーションファイルで指定したパスに生成されます．既に同名のディレクトリが存在する場合は実行を中止します．

  - `Result Directory` - 実行結果を保存します。過去の実行結果は全てここに保存されます．

  - `Database` - aiaccelの内部状態の管理・実行結果を保存するデータベースです．
  work/storage/storage.db に生成されます．
  データベースはsqlite3を採用しています．


## aiaccelの構成モジュール
aiaccelの構成モジュールについて説明します．

![aiaccel_overview](images/aiaccel_modules.png)

- Optimizer
  - 最適化アルゴリズム
    - grid search
    - random
    - sobol sequence
    - nelder-mead
    - tpe
    - mo-tpe
- Scheduler
  - ジョブスケジューラ。`Optimizer` が生成したハイパーパラメータを元にジョブを生成し、計算ノードにジョブを投入します．



## aiaccelの処理フロー
aiaccelの処理フローを説明します。

![aiaccel_flow](images/aiaccel_flow.png)

1. aiaccel-startコマンドからコンフィグを入力として指定して実行します．
2. start.pyがコンフィグをロードし，Masterを起動します．
3. MasterがOptimizerを起動します．
4. MasterがSchedulerを起動します．
5. Optimizerはコンフィグからハイパーパラメータを読み込み，最適化アルゴリズムに基づきハイパーパラメータを生成しStorageに保存します．
6. SchedulerはStorageから新しいハイパーパラメータを読み込み，コンフィグに基づき指定の計算ノードでユーザープログラムをジョブとして実行します．
7. aiaccelのラッパーにより実行されたユーザープログラムが終了すると，aiaccelラッパーがユーザープログラムの結果をStorageに保存します．
8. 5-7 が指定のトライアル数まで繰り返されます．ハイパーパラメータの生成数や同時に実行できる計算ノード数などは全てコンフィグに記述します．
9. 全てのトライアル数分のハイパーパラメータが完了する，または停止命令を受けるとMaster, Optimizer, Scheduler は停止します．



## コードから見るaiaccelの処理フロー
<!-- aiaccelの処理フローでは，大まかにaiaccelではMaster, Optimizer, Schedulerが協調し，それぞれの役割を果たしていることについて述べた．
では実際にコードレベルで，それらのフローを追ってみよう． -->

1. start.py

aiaccelは `aiaccel-start` コマンドで実行を開始します． `aiaccel-start`は、`aiaccel/cli/start.py` を実行します．


Optimizer、Scheduler、の初期化は以下のコードで行われます。

```python
    Optimizer = create_optimizer(args.config)
    Scheduler = create_scheduler(args.config)
```

初期化されたモジュールは，以下のコードで実行される．
pre_processメソッドの後メインループが周り，メインループ後にpost_processメソッドが実行される．
シンプルに表せば基本的にMasterもOptimizerもSchedulerは，これらの処理で説明できる．

```python
    for module in modules:
        module.pre_process()

    while True:
        for module in modules:
            if not module.inner_loop_main_process():
                break
            if not module.check_error():
                break
            module.loop_count += 1
        else:
            time.sleep(sleep_time)
            continue
        break

    for module in modules:
        module.post_process()
```

2. module.py

pre_processメソッド・メインループ・post_processメソッドの基本的な記述は aiaccel/module.py にある．
module.py は，Master, Optimizer, Scheduler のスーパークラスにあたる AbstractModule クラスが定義されている．

3. Master

再度 aiaccel/cli/start.py を見てみる．
Masterモジュールは create_master メソッドにより初期化されている．
aiaccel/master/create.py を見てみると，コンフィグに記述されたresource_typeに基づき異なるMasterクラスが実行される．

簡単のため，ここでは LocalMaster クラスを見てみる．
aiaccel/master/local_master.py を見てみると，AbstractMasterクラスを継承しており特に追記はない．

では更に aiaccel/master/abstract_master.py の AbstractMaster クラスを見てみる．
時間に関するコードや Evaluator などがあるが，inner_loop_main_process メソッド内の以下のコードが終了判定をしている．

```python
        if self.hp_finished >= self.trial_number:
            return False
```

AbstractMaster クラスにおいては，ここで False が返る，つまり終了したハイパーパラメータ数がトライアル数以上になるとMasterが終了する．

4. Optimizer

Optimizerモジュールも，Master同様 start.py にて create_optimizer メソッドにより初期化されている．
aiaccel/optimizer/create.py を見てみると，コンフィグに記述された最適化アルゴリズム名に基づきOptimizerを初期化している．

ここでは簡単のため RandomOptimizer クラス を見てみる．
aiaccel/optimizer/random_optimizer.py を見てみると，AbstractOptimzier クラスを継承しており，generate_parameter メソッドのみオーバーライドされている．

RandomOptimizer クラスの generate_parameter メソッドは，以下のコードでランダムなハイパーパラメータを生成する．

```python
        sample = self.params.sample(rng=self._rng)
```

では更に aiaccel/optimizer/abstract_optimizer.py の AbstractOptimizer クラスを見てみる．
メインループである inner_loop_main_process メソッドを見ると，以下のコードで新しいハイパーパラメータを生成している．

```python
        for _ in range(pool_size):
            new_params = self.generate_new_parameter()
```

pool_size 変数は，計算ノードがどの程度空いているかに基づいた数値である．

5. Scheduler

Schedulerモジュールも，Master, Optimizer 同様のアーキテクチャとなっている．
ここでは LocalScheduler クラスを見てみる．

aiaccel/scheduler/local_scheduler.py は，AbstractScheduler クラスを継承している．
get_stats メソッドは，現在のジョブの状態を取得する役割を担う．
LocalSchedulerクラスでは，ps コマンドをパースしてジョブの状態を取得していることが分かる．

inner_loop_main_process メソッドはメインループであり，ジョブをプロセスとして実行する．
その際の execute メソッドが実行コマンドを生成し実行する．
