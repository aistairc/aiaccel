# aiaccelアーキテクチャ

aiaccelは，与えられた入力ハイパーパラメータ群から最適なハイパーパラメータを出力するハイパーパラメータ最適化ライブラリです．

本章は，開発者に向けたドキュメントです．
aiaccelのアーキテクチャやトピックごとにaiaccelの機能を解説し，aiaccelに関する開発の一助となることを目的としています．

## aiaccelのシステム概要

aiaccelのシステムについて概説します．
aiaccelは，ABCI上で実行することを想定したハイパーパラメータ最適化ライブラリです．
ローカルコンピュータでも動作はしますが，その機能を最大限に発揮するためにはABCIを利用することが推奨されます．
ABCIについては，[ABCI User Guide](https://docs.abci.ai/ja/)を参照してください．
aiaccelは，ABCIのインタラクティブノード上で実行されることを想定されます．
Configを入力として，aiaccelは内部でMaster, Optimizer, Scheduler を起動し，Storage(ファイルシステムやデータベース)に状態を保存しながら，最適化対象であるユーザープログラムをABCI計算ノードにてジョブとして実行します．
ABCI計算ノードで実行されたユーザープログラムは，結果をStorageに保存します．

![aiaccel_system](images/aiaccel_system.png)

## aiaccelの入出力

aiaccelの入出力をもう少しく詳しくみてみます．
- 入力
  - Config: コンフィグレーションファイルです．
  examples/sphere/config.yaml などが該当します．
  上記のシステム概要のConfigの一部です．
  - User Program: ユーザープログラムです．
  examples/sphere/user.py などが該当します．
  user.py は上記のコンフィグレーションファイル内で指定します．
  user.pyである必要はありませんが，aiaccelがユーザープログラムを実行するためのインタフェースが user.py に記述されていますので user.py に該当するファイルは必要となります．
  Java, C++ などの実行ファイルを利用する場合は user.py に該当するファイルから呼び出して実行してください．
  上記のシステム概要の\<user program>.pyです．
  - Job Script: ジョブスクリプトファイルです．
  examples/sphere/job_script_preamble.sh などが該当します．
  ジョブスクリプトファイルは，ABCIを利用する際に必要となるファイルです．
  役割は，aiaccelがABCI上でuser.pyをジョブとして実行するためのスクリプトとなります．
  詳しくは[ABCIのバッチジョブに関するドキュメント](https://docs.abci.ai/ja/job-execution/)の記法を参照してください．
  上記のシステム概要のConfigの一部です．
- 出力
  - Work Directory: aiaccelを実行した際生成されるワークディレクトリです．
  aiaccelを実行した際 work という名前のディレクトリが生成されます．
  上記のシステム概要のStorageの一部です．
  - Result Directory: aiaccelを実行した際，実行結果を保存するリザルトディレクトリです．
  ワークディレクトリは，現在実行中・実行した状態を保存するディレクトリであるのに対し，リザルトディレクトリは過去に実行した結果を全て保存するディレクトリです．
  ただし，実行したディレクトリ内に生成されるため，実行するディレクトリを変更するとまた新しいリザルトディレクトリが生成されます．
  上記のシステム概要のStorageの一部です．
  - Database: aiaccelの実行中の状態・実行結果を保存するデータベースです．
  work/storage/storage.db が該当します．
  work はワークディレクトリです．
  データベースはsqlite3を採用しています．
  上記のシステム概要のStorageの一部です．

![aiaccel_input_output](images/aiaccel_input_output.png)

## aiaccelの構成モジュール
aiaccelは，内部で３つのモジュールが連携しながら実行されます．
本節ではaiaccelの３つのモジュールの役割について説明します．

- マスター
    - スケジューラ・オプティマイザを管理します．
    開始時に起動され，オプティマイザ・スケジューラを起動し，これら２つのモジュールの死活監視をします．
    オプティマイザ(またはスケジューラ)が停止すると実行中のスケジューラ(またはオプティマイザ)を停止させ，自身も終了します．
- オプティマイザ
    - どのハイパーパラメータを次に実行するかを計算します．５つの最適化アルゴリズムをサポートしており，コンフィグに記述することで実行するアルゴリズムを選択します．
- スケジューラ
    - オプティマイザが選択したハイパーパラメータをジョブとして実行し，そのジョブを管理します．
    ジョブは，ハイパーパラメータごとに生成されローカルコンピュータまたはABCI上で実行されます．

![aiaccel_overview](images/aiaccel_modules.png)

## aiaccelの処理フロー
aiaccelが内部でどのように実行されるかを別の視点から見てみます．
以下の図でもマスター・オプティマイザ・スケジューラの３つのモジュールを軸に構成されています．

1. aiaccel-startコマンドからコンフィグを入力として指定して実行します．
2. start.pyがコンフィグをロードし，Masterを起動します．
3. MasterがOptimizerを起動します．
4. MasterがSchedulerを起動します．
5. Optimizerはコンフィグからハイパーパラメータを読み込み，最適化アルゴリズムに基づきハイパーパラメータを生成しStorageに保存します．
6. SchedulerはStorageから新しいハイパーパラメータを読み込み，コンフィグに基づき指定の計算ノードでユーザープログラムをジョブとして実行します．
7. aiaccelのラッパーにより実行されたユーザープログラムが終了すると，aiaccelラッパーがユーザープログラムの結果をStorageに保存します．
8. 5-7 が指定のトライアル数まで繰り返されます．ハイパーパラメータの生成数や同時に実行できる計算ノード数などは全てコンフィグに記述します．
9. 全てのトライアル数分のハイパーパラメータが完了する，または停止命令を受けるとMaster, Optimizer, Scheduler は停止します．

![aiaccel_flow](images/aiaccel_flow.png)

## コードから見るaiaccelの処理フロー
aiaccelの処理フローでは，大まかにaiaccelではMaster, Optimizer, Schedulerが協調し，それぞれの役割を果たしていることについて述べた．
では実際にコードレベルで，それらのフローを追ってみよう．

1. start.py

aiaccelはaiaccel-startスクリプトにより実行を開始する．aiaccel/cli/start.py を見てみるとまずMaster, Optimizer, Schedulerが初期化される

```python
    Master = create_master(args.config)
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
