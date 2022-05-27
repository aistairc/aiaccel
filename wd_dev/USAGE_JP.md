<!-- -*- coding: utf-8-unix; mode: Text; -*- -->

<!-- ----- ----- ----- --><a id="目次"></a>
# 目次

- [aiaccelのインストール](#aiaccelのインストール)
  1. ~/opt のダウンロード直後。
  2. インストール。「~/opt/README_JP.md」通りに
  3. テスト
- [wdの動作の仕組み](#wdの動作の仕組み)
  1. wdの動作の概要
  2. フォルダ構造について
- [wdのインストール](#wdのインストール)
  1. 前提
  2. インストール
  3. テスト
     1. フォルダを移動
     2. ABCIのグループ名を設定
     3. CUIにより実行開始を指示
- [sample ai(評価関数)のインストール](#sample-ai評価関数のインストール)
  1. TensorFlow-Kerasとkeras-io/examples/vision/mnist_convnet.pyのインスール
  2. 01-1grid-1gpu/000-grid1/mnist_convnet.shのテスト
  3. 01-1grid-1gpu/000-grid1/user.pyのテスト
  4. aiaccel(abci mode)からのテスト
  5. wdからのテスト
- [wdの使用事例-2つのaiaccel](#wdの使用事例-2つのaiaccel)
  1. 概要
  2. 実行
     1. 実行経過
     2. 計算ノード上でpythonプログラムnvidia-smi.pyを実行
     3. 実行結果
- [wdのデバッグ事例](#wdのデバッグ事例)
  1. qsubを実行したふりをするデバッグモードtest_no_qsubの設定方法
  2. PYTHONPATHを設定するmake_wd_env_for_dev.source
  3. 1つのaiaccelで多数のtrial
  4. 8つのaiaccel
- [wdのその他のコマンドなど](#wdのその他のコマンドなど)
  1. ndへのプログラムrequest用のpython -m wd.bin.nreq
- [wdの試作状況](#wdの試作状況)

<!-- ----- ----- ----- --><a id="aiaccelのインストール"></a>
# aiaccelのインストール
[目次へ](#目次)

## 1. ~/opt のダウンロード直後。

以下、
```
ssh abci
```
により、abciのインタラクティブノードにlogin後の
実際に入力するコマンドを示しながら、ご説明いたします。  

申しわけありませんが、
いわゆるlinuxのコマンドやpythonなどに対するある程度のスキルを前提にいたします。  

インタラクティブノードへのlogin方法については、
「ABCI 2.0 User Guide」の
[「ABCIの利用開始」](https://docs.abci.ai/ja/getting-started/)の
「SSHクライアントによるログイン」
をご参照いただきたく。
```
なお、git cloneなどにより、~/opt が作成されていることを前提にいたします。
作業時に、~/keras-io, ~/optenv, ~/tfenv (および ~/.keras など)を作成いたします。

ssh abci
cd ~/opt
```
インタラクティブノードでの作業は、課金されていません。
そのため、abci使用準備などの最小限の作業のみに限定とのことです。
[「ABCIシステムの概要」](https://docs.abci.ai/ja/system-overview/)に、下記の記述がございます。
> Warning
> インタラクティブノードのCPUやメモリなどの資源は多くの利用者で共有するため、高負荷な処理は行わないようにしてください。高負荷な前処理、後処理を行う場合は、計算ノードを利用してください。 インタラクティブノードで高負荷な処理を行った場合、システムにより処理が強制終了されますのでご注意ください。

## 2. インストール。[「~/opt/README_JP.md」](../README_JP.md)通りに
abciのmodule関連の記述などの詳細は、
[「Environment Modules</a>」](https://docs.abci.ai/ja/environment-modules/)も合わせてご参照いただきたく。  
~/opt/wd_dev/tools/[install_opt.source](tools/install_opt.source)  
```
source ~/opt/wd_dev/tools/install_opt.source
```
3分程、時間がかかります。

```
abciのmoduleは常に更新されているため、執筆時点のmoduleが無い場合もございます。
その場合は、エラーとなりますので、
大変、お手数ではございますが、
module available
で最新のmoduleを確認後、install_opt.sourceの
module load gcc/11.2.0
module load python/3.8/3.8.13
の部分をご変更いただきたく。
(~/opt/wd_dev/toolsの中の他のファイルも同様にお願いいたしたく。)
```

出力の最後が、
```
Finished processing dependencies for aiaccel==0.0.4
```
であれば、正常です。

念のため、importできるとことを確認します。
```
cd ~
python
import aiaccel
exit()
```

## 3. テスト

1. ~/opt/wd_dev/examples/config/opt-test/readme_jp-sample-local/[config.yml](examples/config/opt-test/readme_jp-sample-local/config.yml)  
README_JP.mdの「チュートリアル」の「2. コンフィグファイルの作成」の
「コンフィグファイル サンプル」のconfig.ymlをコピペして作成したファイル。
2. ~/opt/wd_dev/examples/config/opt-test/readme_jp-sample-local/[user.py](examples/config/opt-test/readme_jp-sample-local/user.py)  
「3 ユーザープログラムの作成」のサンプルをコピペして作成したファイル。

上記2つのファイルのあるフォルダに移動後に実行。
テスト時の実行時間は、5分程でした。
インタラクティブノードで実行しているため、課金はされません。
何度もテストしたい場合は、計算ノードを使っていただきたく。

```
cd ~/opt/wd_dev/examples/config/opt-test/readme_jp-sample-local/
python -m aiaccel.start --config config.yml --clean --graph
```
出力の最後が、
```
moving...
Best result    : results/20211220_085512/hp/finished/000029.hp
               : -6.999000256189666
Total time [s] : 269
```
であれば、正常です。
数字の部分は若干違うかも知れません。

インストール後は、

~/opt/wd_dev/tools/[make_opt_env.source](tools/make_opt_env.source)  
```
ssh abci
source ~/opt/wd_dev/tools/make_opt_env.source
cd ~/opt/wd_dev/examples/config/opt-test/readme_jp-sample-local/
python -m aiaccel.start --config config.yml --clean --graph
```
で、実行できます。

計算ノードを使用する場合は、
```
qrsh -g gaa*****(ご自身のグループ名) -l rt_C.small=1 -l h_rt=1:00:00
```
で、計算ノードにログイン後、計算機が変わったので、再度、
```
source ~/opt/wd_dev/tools/make_opt_env.source
cd ~/opt/wd_dev/examples/config/opt-test/readme_jp-sample-local/

```
していただき、
```
python -m aiaccel.start --config config.yml --clean --graph

```
で、実行します。テスト時は、

作業が終わりましたら、すぐに、
```
exit
```
していただきたく。

exit後、念のため、
```
qstat
```
の出力に表示されないことで、qrshの終了をご確認いただきたく。

ポイントは、執筆時点で、1時間当たり0.2ポイントです。
GPUを使用しないため、rt_C.smallを使用しています。

参考資料: [ジョブ実行](https://docs.abci.ai/ja/job-execution/) ABCI 2.0 User Guide

<!-- ----- ----- ----- --><a id="wdの動作の仕組み"></a>
# wdの動作の仕組み
[目次へ](#目次)

## 1. wdの動作の概要

1. local mode(単一PC)のaiaccelは、
ai(評価関数)としてuser.pyを子プロセスとして実行し、
計算結果を受けとります。
aiaccelにはabci mode(abciの計算ノードを使用)も用意されていますが、
wdは、local modeで動作させます。

2. wdは、aiaccelがuser.pyを子プロセスとして実行する前にフックします。

3. フック後、abciの計算ノードでuser.pyを実行し、その結果をフック時点に返します。

4. aiaccelはフックされたことは認識せずに、通常通り、子プロセスから、結果を受けとります。

以上の仕組みのため、local modeのaiaccelは、裏でabciの計算ノードが実行されていることを認識していません。
同じPC上で、子プロセスとして実行されているものと認識しています。

wdは、aiaccelが下記のフォルダ構造のconfig.ymlから実行された時にのみ、フックします。
```
nn-*/nnn-*/config.yml (nは1桁の数字。*は任意の文字列)
```

## 2. フォルダ構造について

1. nn-\*/について。  
将来、wd自身を同時に複数実行できるようにする予定です。
(今でも手作業により、同時に複数実行することは可能です。)
そのため、nn-\*/というフォルダ構造を前提にしました。
nnは、2桁の数字であれば、いくつでも問題はありません。
nn-\*/フォルダの中に、1つのwdの動作の設定などを記述します。

2. nnn-\*/(nn-\*/の中の)について。  
wdは、同時に複数のaiaccelを実行できます。
そのため、nnn-\*/というフォルダ構造を前提にしました。
nnn-\*/フォルダの中に、1つのaiaccelの動作の設定などを記述します。
大規模実験時には、100以上のaiaccelを動かす可能性があるため、3桁にしました。
今回のものは試作であり、大規模、長期運用テストはしておりません。
大変、申しわけありませんが、examplesで何とか動作し、
wdの概要を把握していただくレベルとお考えいただきたく。
(nnnは、000から通番の3桁の数字。)

<!-- ----- ----- ----- --><a id="wdのインストール"></a>
# wdのインストール
[目次へ](#目次)

## 1. 前提

[aiaccelのインストール](#aiaccelのインストール)の通り、
既にaiaccelがインストールされている状態を前提に、
ご説明いたします。

実は、wdはopt(aiaccelをgit cloneする時の名前)のダウンロード時に、
```
~/opt/wd_dev
```
に、既にダウンロードされていますが、
試作中の機能のため、aiaccelのインストール時には、インストールされません。  
aiaccelのソースに、wdへフックする部分は埋め込まれていますが、
wdをインストール後に、前述した条件がそろわないと、wdは動作しません。

## 2. インストール

以下のコマンドを実行すると、
aiaccelのインストール時に作成した
aiaccel用のvenv環境である
```
~/optenv
```
に、wdをインストールします。

~/opt/wd_dev/tools/[install_wd.source](tools/install_wd.source)  
```
ssh abci
source ~/opt/wd_dev/tools/install_wd.source
```
出力の最後が、
```
Finished processing dependencies for wd==0.0.1
```
であれば、正常です。

念のため、importできることを確認します。
```
cd ~
python
import wd
exit()
```

## 3. テスト

### 1. フォルダを移動

```
cd ~/opt/wd_dev/examples/config/00-1grid-0gpu
```
現状では、完全な初期状態の時にのみ実行できます。
そのため、初回は必要有りませんが、実行時に作成されるフォルダを削除しておきます。
-fオプション(強制実行)を使用しているため、慎重に実行していただきたく。
```
rm -rf work 000-grid1/work
```

### 2. ABCIのグループ名を設定

~/opt/wd_dev/examples/config/00-1grid-0gpu/000-grid1/[config.yml](examples/config/00-1grid-0gpu/000-grid1/config.yml)の
```
ABCI:
  group: "[group]"
```
の部分の
group
を例えば
```
ABCI:
  group: "[gaa*****]"
```
のようにご自身のグループ名に変更して下さい。

wdは、000-\*/config.yml(通番が最初の000)のグループ名のみを参照します。
今回はありませんが、001-\*以降のconfig.ymlのグループ名は参照しないため、そのままでも問題ありません。

### 3. CUIにより実行開始を指示

実行時に、rt_G.smallの計算ノード1つを1分程使用します。
(執筆時rt_G.smallは1時間当たり0.3ポイント消費します。)

以下のコマンドを入力し、実行します。
```
python -m wd.bin.cui
```
試作中のため、メニューが簡易すぎ、使用しづらく申しわけありません。
```
0.プログラムを終了
1.aiaccelの実行
2.workの閲覧など
3.コマンド(ps,qstat,qdel-all,nreq-ls,nvidia-smiなど)
4.wd.bin.qsubの実行、停止、動作確認
5.*-wd/workの削除
数字を入力
```
キーボードより数字の1を入力(1.aiaccelの実行)。
```
0.前のメニューに戻る
1.実行方法の簡単な説明
2.計算ノード数の確認
3.実行
数字を入力
```
3を入力(3.実行)。

```
aiaccelの実行を開始します。

完全な初期状態で、実行して下さい。
今のところ、前回のデータの有無などのチェックはしていません。
そのため、データや実行中のプロセスなどがあると、誤動作します。
(チェック機能は、作成中です。)

よろしいですか。

1.はい
その他のキー.いいえ
```
1を入力(1.はい)。

```
wd経由のaiaccelの計算ノードでの実行を開始後、
本プログラムを終了します。
必要な場合は、再度、
本プログラム(python -m wd.bin.cui)を実行して、
途中経過を確認できます。

任意のキーで次へ
```
0を入力(任意ですが)。

コンソールへの出力が見難くなるため、
aiaccelの実行開始の指示の後、
プログラムを終了します。

終了前に、

```
Cui       INFO     start
Cui       DEBUG    wd.bin.qsub
wd.bin.qsubを実行しました。
Cui       DEBUG    wd.bin.qreq -s
Qreq      INFO     start
Qreq      INFO     q0000-request
Qsub      INFO     start

q0000のnd待ち !!!

.Qsub      INFO     qsub -g グループ名 -o /home/ユーザ名/opt/wd_dev/examples/config/00-1grid-0gpu/work/qsub/output/q0000 /home/ユーザ名/opt/wd_dev/examples/config/00-1grid-0gpu/work/qsub/q0000-running
Qsub      INFO     Your job 8526575 ("q0000-running") has been submitted
```
と表示されます(job番号の部分などは多少違います)。
試作中のため、表示される情報が多くて申しわけありません。

この後に、q0000(qsubのノード名の最初の5文字)計算ノードの実行と、その後のnd(コマンド受付用プログラム)の実行を待ちます。
ndが起動されると、
```
.....!
wd.bin.qreqを実行しました。
Cui       DEBUG    wd.bin.nreq q0000 gpu
Nreq      INFO     start
Nreq      INFO     run_gpu.py-000-request
Cui       DEBUG    wd.bin.nreq -o 0 q0000 opt
Nreq      INFO     start
Nreq      INFO     run_aistpot.py-000-request
Cui       DEBUG    wd.bin.nreq q0000 ai
Nreq      INFO     start
Nreq      INFO     run_ai.py-000-request
wd.bin.nreqを実行しました。

aiaccelの計算ノードでの実行を開始しました。
途中経過は、再度、
python -m wd.bin.cui
を実行することにより確認できます。
そのため、abciからログアウトしていただいても問題有りません。
本プログラム(python -m wd.bin.cui)を終了します。!!!
```
と表示されます。

1分程たつと、計算ノードでの計算が終わり、
```
Cui       INFO     end
ユーザ名 has registered the job 8526575 for deletion
Qsub      INFO     stop_all_qsub: qdel 8526575
Qsub      INFO     optが全て終了しました。
全てのqsubを終了します。
本daemonも終了します。
```
と表示されます。
コマンドプロンプトが表示されていない場合は、
Enterキーを入力していただきたく。

正常に終了したことを確認します。
```
python -m wd.bin.cui
```
2を入力(2.workの閲覧など)。
```
0.前のメニューに戻る
1.*-wd/work閲覧。treeで
2.*-wd/work/log閲覧
3.*-wd/work閲覧
4.nreq/tmp閲覧
5.opt閲覧
数字を入力
```
1を入力(1.*-wd/work閲覧。treeで)。
```
(cd /home/ユーザ名/opt/wd_dev/examples/config/00-1grid-0gpu/work; tree -an)
1.前へ 2.次へ 3.最後へ 4.左へ 5.右へ 0.戻る
000|.
001|├── gpu
002|│   └── 000-grid1
003|│       └── 0000-q0000-g0-finished
004|├── log
005|│   ├── ai_q0000.log
006|│   ├── cui.log
007|│   ├── gpu_q0000.log
008|│   ├── nd_q0000.log
009|│   ├── nreq.log
010|│   ├── opt_000.log
011|│   ├── qreq.log
012|│   └── qsub.log
013|├── node
014|│   └── q0000
015|│       ├── run_ai.py-000-running
016|│       ├── run_aistpot.py-000-finished
017|│       └── run_gpu.py-000-running
018|├── qsub
019|│   ├── output
020|│   │   └── q0000
021|│   └── q0000-running
022|└── var
023|
024|8 directories, 14 files
025|
```
で、003に、0000-q0000-g0-finishedがあれば、
正常に終了しています。

表示の意味などは、後述の[wdの使用事例-2つのaiaccel](#wdの使用事例-2つのaiaccel)でご説明いたします。

0を入力(0. 戻る)。

0を入力(0. 前のメニューに戻る)。

0を入力(0. プログラムを終了)。

で、終了できます。

正常に終了されていれば、問題無いのですが、
念のため、
```
qstat
```
により、実行中の計算ノードが無いことを確認していただきたく。

覚えの無い計算ノードが有りましたら、
大変お手数ではございますが、
```
python -m wd.bin.cui
```
の
```
0.プログラムを終了
1.aiaccelの実行
2.workの閲覧など
3.コマンド(ps,qstat,qdel-all,nreq-ls,nvidia-smiなど)
4.wd.bin.qsubの実行、停止、動作確認
5.*-wd/workの削除
数字を入力
```
3を入力(3.コマンド(ps,qstat,qdel-all,nreq-ls,nvidia-smiなど))。
```
0.前のメニューに戻る
1.ps
2.qstat
3.qdel all
4.nreq nvidia-smi.py q0000
5.nreq ps.py q0000
6.nreq ls.py q0000
数字を入力
```
3を入力(3.qdel all)。

により、wd関連の計算ノードを全て終了していただきたく。

既に全て終了している時は、
```
qdel_all: qstat: 標準出力は空でした。
任意のキーで次へ
```
と表示されます。

執筆時点でのテスト時には問題が起きませんでしたが、
念のため、対応方法をご説明いたしました。

インストール後は、

~/opt/wd_dev/tools/[make_wd_env.source](tools/make_wd_env.source)  
```
ssh abci
source ~/opt/wd_dev/tools/make_wd_env.source
cd ~/opt/wd_dev/examples/config/00-1grid-0gpu
rm -rf work 000-grid1/work
python -m wd.bin.cui
```
で、実行できます。
wdインストール後は、  
make_wd_env.sourceのみの実行で問題ありません。  
make_opt_env.sourceを合わせて実行する必要はありません。

<!-- ----- ----- ----- --><a id="sample-ai評価関数のインストール"></a>
# sample ai(評価関数)のインストール
[目次へ](#目次)

## 1. TensorFlow-Kerasとkeras-io/examples/vision/mnist_convnet.pyのインスール

wdのインストール時には、
GPUを使用しない評価関数を使用して、
テストを行いました。

後述のwdの実行例では、GPUを使用した場合について、ご説明します。
そのため、その時に使用するsample ai(評価関数)のインストールを行います。

aiaccelや試作中のwdを使用する時に、
ご自身のai(評価関数)を使用されると思いますが、
その時の対応方法などの参考にしていただきたく。

sample ai(評価関数)として、
abciのGPU使用時の実行時間が1分強程の
kerasのmnist_convnet.py
を使用いたします。

ご自身のai(評価関数)を使用する場合は、
この部分を置き換えることになります。

「ABCI 2.0 User Guide」にも
「[TensorFlow-Keras](https://docs.abci.ai/ja/apps/tensorflow-keras/)」として説明がありますので、
その記述にそってインストールします。
(下記のqrshは課金されます。)

```
ssh abci
qrsh -g ご自身のグループ名 -l rt_G.small=1 -l h_rt=1:00:00
```
で、GPU1つの計算ノードにログイン。

~/opt/wd_dev/tools/[install_keras_io.source](tools/install_keras_io.source)  

```
source ~/opt/wd_dev/tools/install_keras_io.source
```
で、インストール。時間は、2分程でした。
```
python3 ~/keras-io/examples/vision/mnist_convnet.py
```
1分強で終了(1回目のテストの時)。
1回目は、mnistデータをダウンロードするため、多少時間が多めになります。
```
exit
```
実験終了後は、課金を最小にするために、qrshから、すぐにexitすることを、お勧めいたします。
念のため、課金が終了したことを確認。
```
qstat
```
課金(ポイント)の状況の確認。
```
show_point
```
出力結果の最後を掲載。数値は多少変わります。
```
Test loss: 0.025734134018421173
Test accuracy: 0.9912999868392944
```
mnistデータの確認。
```
ls -l ~/.keras/datasets
```

## 2. 01-1grid-1gpu/000-grid1/mnist_convnet.shのテスト

~/opt/wd_dev/examples/config/01-1grid-1gpu/000-grid1/[mnist_convnet.py](examples/config/01-1grid-1gpu/000-grid1/mnist_convnet.py)

~/keras-io/examples/vision/mnist_convnet.pyとのdiff
```
12a13
> import sys  # wd/
55c56
<         layers.Dropout(0.5),
---
>         layers.Dropout(float(sys.argv[-3])),  # wd/ 0.5
66,67c67,68
< batch_size = 128
< epochs = 15
---
> batch_size = int(sys.argv[-2])  # wd/ 128
> epochs = int(sys.argv[-1])  # wd/ 15
```

~/opt/wd_dev/examples/config/01-1grid-1gpu/000-grid1/[mnist_convnet.sh](examples/config/01-1grid-1gpu/000-grid1/mnist_convnet.sh)

テスト。
```
ssh abci
qrsh -g ご自身のグループ名 -l rt_G.small=1 -l h_rt=2:00:00
source ~/opt/wd_dev/tools/make_wd_env.source
cd ~/opt/wd_dev/examples/config/01-1grid-1gpu/000-grid1
./mnist_convnet.sh 0.5
exit
qstat
```
出力結果の最後を掲載。
```
Epoch 15/15
422/422 [==============================] - 1s 2ms/step - loss: 0.0330 - accuracy: 0.9892 - val_loss: 0.0307 - val_accuracy: 0.9925
Test loss: 0.02569478377699852
Test accuracy: 0.9912999868392944
0.02569478377699852(optenv) -bash-4.2$
```
最終行の数値を、後述のuser.pyが評価関数の戻り値として使用します。
改行していないため、直後にbashのプロンプトが出力されています。
数値は多少違います。

## 3. 01-1grid-1gpu/000-grid1/user.pyのテスト

~/opt/wd_dev/examples/config/01-1grid-1gpu/000-grid1/[user.py](examples/config/01-1grid-1gpu/000-grid1/user.py)

環境を整えます。
```
ssh abci
qrsh -g ご自身のグループ名 -l rt_G.small=1 -l h_rt=2:00:00
source ~/opt/wd_dev/tools/make_wd_env.source
cd ~/opt/wd_dev/examples/config/01-1grid-1gpu/000-grid1
rm -rf work
```
2分程、何も出力しないため、backgroundで実行します。
```
python user.py --index 000000 --config config.yml --x1=0.1 &
```
動いているかどうか不安なので、
```
tree work
```
で、

```
work
├── lock
│   └── result
└── log
    └── wrapper.log

2 directories, 2 files
```
を確認します。
2分程待つと、

```
(optenv) -bash-4.2$ objective_y:0.03315233439207077
Traceback (most recent call last):
(中略)
FileNotFoundError: [Errno 2] No such file or directory: 'work/result/000000.result'
```
と出力されますが、テストは正常に終了しています。数値の部分は多少違います。
```
exit
qstat
```
で、qrshを終了します。

## 4. aiaccel(abci mode)からのテスト

aiaccel(abci mode)からのテストは省略します。  
wdでは必要の無い、aiaccelのためのABCI設定、job_script_preamble.shの作成などが必要なため。

## 5. wdからのテスト

[wdのインストール](#wdのインストール)と同様に、
以下のコマンドを入力して、テストします。

テスト時に忘れやすい点。  
~/opt/wd_dev/examples/config/01-1grid-1gpu/000-grid1/[config.yml](examples/config/01-1grid-1gpu/000-grid1/config.yml)の  
ABCIのグループ名の変更。

config.ymlで、
>   type: "ABCI"

となっていますが、そのままで大丈夫です。
wd経由の場合、設定にかかわらず、typeはlocalで実行します。

```
ssh abci
source ~/opt/wd_dev/tools/make_wd_env.source
cd ~/opt/wd_dev/examples/config/01-1grid-1gpu
rm -rf work 000-grid1/work
python -m wd.bin.cui
```
の後、メニューを選択して、テスト実行します。

まだ使用していないメニューをご紹介します。  
1を入力(1. aiaccelの実行)  
2を入力(2.計算ノード数の確認)
```
aiaccelの数: 1
num_nodeの合計: 1
trial_numberの合計: 1
計算ノードの合計: 1
G.large: 0
G.small: 1

計算ノードの合計数が大きすぎるなどの妥当性のチェックはしていません。
大きすぎる場合は、aiaccelのnum_nodeで調整して下さい。

G.largeは、71時間で終了します。
G.smallは、167時間で終了します。
再実行は実装確認中です。
spotのG.largeは72時間で、G.smallは168時間で、abciから強制終了されます。

一時的にpointを消費するため、小規模なテストなどの時は、
*-wd/config.ymlの
wd.max_large_minutes 現在の値: 1*60
wd.max_small_minutes 現在の値: 1*60
で終了までの時間を短くすることができます。

任意のキーで次へ
```
で、使用する計算ノード数を確認できます。
> aiaccelの数: 1

000-grid1フォルダしか無いため。01-1grid-1gpuフォルダの中に。
> num_nodeの合計: 1

000-grid1/[config.yml](examples/config/01-1grid-1gpu/000-grid1/config.yml)のnum_nodeの設定が1のため。

> trial_numberの合計: 1

config.ymlのtrial_numberの設定が1のため。
> 計算ノードの合計: 1  
G.large: 0  
G.small: 1  

num_nodeの合計が1なので、計算ノードは、G.small(GPU1つ)が1つのみ。

この後の実行方法などは前述と同様。

<!-- ----- ----- ----- --><a id="wdの使用事例-2つのaiaccel"></a>
# wdの使用事例-2つのaiaccel
[目次へ](#目次)

## 1. 概要

1GPUの計算ノードG.smallが4つより、4GPUの計算ノードG.largeが1つの方が、
GPUは同じ4つですが、spotのポイントは25%安くできます。

そのため、G.largeを使うべきですが、
optimizerによっては、GPUの使用状況にむらができてしまい、
使用せずに遊んでいるGPUが発生し、無駄が生じます。

その問題を解決するために、
1. 複数のaiaccelを同時に動かす。
2. 各aiaccelのnum_node数を全て合算した値を全てのaiaccelのnum_nodeとする。

という単純なルールで動作させることにしました。

本事例(~/opt/wd_dev/examples/config/02-1grid-1nm-8gpu/)では、
> grid search  
num_node 4  
trial_number 24

> nelder-mead search  
num_node 4  
trial_number 8

という2つのaiaccelを、wdが、同時に、

> grid search  
num_node 8  

> nelder-mead search  
num_node 8  

として、実行します。

並列実行などの工夫をしていない
2パラメータのいわゆるnelder-mead searchは、
最初の3回のみ並列ですが、後は単発で実行することになります。

ですが、grid searchは、それこそ、全て並列実行が可能ですので、
GPUが空いていれば、即、次の計算を実行することができます。

そのため、num_nodeが4では無く、両方とも8であれば、
nelder-mead searchが使用していないGPUをgrid searchが使用することができ、
GPUの使用効率が良くなります。

## 2. 実行

実行時間は、12分程です。

000-grid/[config.yml](examples/config/02-1grid-1nm-8gpu/000-grid/config.yml)

001-nm/[config.yml](examples/config/02-1grid-1nm-8gpu/001-nm/config.yml)

上記は、パラメータのサーチ結果よりもwdの動作説明に重きを置いた設定です。

実行時に忘れやすい点。  
~/opt/wd_dev/examples/config/02-1grid-1nm-8gpu/000-grid/[config.yml](examples/config/02-1grid-1nm-8gpu/000-grid/config.yml)の  
ABCIのグループ名の変更。
こちらのみでok。001-nm/config.ymlはそのままでok。

```
ssh abci
source ~/opt/wd_dev/tools/make_wd_env.source
cd ~/opt/wd_dev/examples/config/02-1grid-1nm-8gpu
rm -rf work 000-grid/work 001-nm/work
python -m wd.bin.cui
```
の後、メニューを選択して、実行します。

実行時のパラメータ。
```
aiaccelの数: 2
num_nodeの合計: 8
trial_numberの合計: 32
計算ノードの合計: 2
G.large: 2
G.small: 0
```

計算ノードが2つ有るため、
```
以下、一部抜粋。
q0000のnd待ち !!!
......!
q0001のnd待ち !!!
.......!
```
と、2つの計算ノード上のndの実行開始を待ってから、処理を開始。

### 1. 実行経過

2を入力(2.workの閲覧など)  
1を入力(1.*-wd/work閲覧。treeで)

```
(cd /home/acb11523fz/opt/wd_dev/examples/config/02-1grid-1nm-8gpu/work; tree -an)
1.前へ 2.次へ 3.最後へ 4.左へ 5.右へ 0.戻る
000|.
001|├── gpu
002|│   ├── 000-grid
003|│   │   ├── 0000-q0001-g0-finished
004|│   │   ├── 0001-q0000-g2-running
005|│   │   ├── 0002-q0001-g3-finished
006|│   │   ├── 0003-q0001-g1-finished
007|│   │   ├── 0004-q0000-g3-finished
008|│   │   ├── 0005-q0000-g3-running
009|│   │   ├── 0006-q0000-g1-running
010|│   │   ├── 0007-q0000-g2-finished
011|│   │   ├── 0008-q0000-g0-running
012|│   │   ├── 0009-request
013|│   │   ├── 0010-q0001-g3-running
014|│   │   ├── 0011-q0001-g2-running
015|│   │   └── 0012-q0001-g1-running
016|│   └── 001-nm
017|│       ├── 0000-q0000-g1-finished
018|│       ├── 0001-q0001-g2-finished
019|│       ├── 0002-q0000-g0-finished
020|│       └── 0003-q0001-g0-running
```

実行経過の見方。
> 002|│   ├── 000-grid

000-gridのoptの実行経過。
> 003|│   │   ├── 0000-q0001-g0-finished

1. request番号が0000。この番号は、000-grid/work/hp/finished/の000000.hpの番号と一致。
2. 計算ノード q0001で実行。
3. g0、0から数えて0番目のGPUで実行。

実行経過で注目して欲しいのは、
1. runningの状態が8つ有ること。gpuが8つあるので。
2. 001-nmにはrequest(実行待ち)が無く、runningが1つだけ。nelder-meadのアルゴリズムにより。
3. 000-gridにはrequestがあること。
   000-gridのrequestとrunningの合計が8なのは、num_nodeが8だから。

### 2. 計算ノード上でpythonプログラムnvidia-smi.pyを実行

wdには、計算ノード上でpythonプログラムを実行する機能が有ります。
その機能を使用して、
~/opt/wd_dev/examples/nreq/nvidia-smi.py
```
import sys
import os

node_name = sys.argv[1]
nn_wd_path = sys.argv[2]

cmd = ('nvidia-smi >> '
       '%s/work/tmp/%s-nvidia-smi.txt' % (nn_wd_path, node_name))
os.system(cmd)

```
を実行して、動作状況を確認。

メニューから実行を指示。  
3を入力(3.コマンド(ps,qstat,qdel-all,nreq-ls,nvidia-smiなど))  
4を入力(4.nreq nvidia-smi.py q0000)
```
python -m wd.bin.nreq q0000 $WD_DEV_DIR/examples/nreq/nvidia-smi.py
1.前へ 2.次へ 3.最後へ 4.左へ 5.右へ 0.戻る
000|Nreq      INFO     start
001|Nreq      INFO     nvidia-smi.py-001-request
002|
```
計算ノードq0000上でのnvidia-smi.pyの実行をrequest。  
nvidia-smi.py-001-requestの001の部分は、0から数えてのこのプログラムのrequest回数。
001、つまり2回目の実行。同じプログラムを何度でも実行できます。

requestを簡便にするために、python -m wd.bin.nreqというコマンドを用意しています。
使用方法などは、後述します。

メニューで出力結果を確認。  
2を入力(2.workの閲覧など)  
3を入力(3.*-wd/work閲覧)  
5を入力(5.tmp  20-19:34:12 2021/12)  
1を入力(1.q0000-nvidia-smi.txt  20-19:34:03 2021/12)
```
/home/ユーザ名/opt/wd_dev/examples/config/02-1grid-1nm-8gpu/work/tmp/q0000-nvidia-smi.txt
1.前へ 2.次へ 3.最後へ 4.左へ 5.右へ 0.戻る
000|Mon Dec 20 19:34:03 2021
001|+-----------------------------------------------------------------------------+
002|| NVIDIA-SMI 470.57.02    Driver Version: 470.57.02    CUDA Version: 11.4     |
003||-------------------------------+----------------------+----------------------+
004|| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
005|| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
006||                               |                      |               MIG M. |
007||===============================+======================+======================|
008||   0  Tesla V100-SXM2...  On   | 00000000:3D:00.0 Off |                    0 |
009|| N/A   28C    P0    56W / 300W |  15462MiB / 16160MiB |      0%      Default |
010||                               |                      |                  N/A |
011|+-------------------------------+----------------------+----------------------+
012||   1  Tesla V100-SXM2...  On   | 00000000:3E:00.0 Off |                    0 |
013|| N/A   26C    P0    58W / 300W |  15462MiB / 16160MiB |      0%      Default |
014||                               |                      |                  N/A |
015|+-------------------------------+----------------------+----------------------+
016||   2  Tesla V100-SXM2...  On   | 00000000:B1:00.0 Off |                    0 |
017|| N/A   25C    P0    58W / 300W |  15462MiB / 16160MiB |      0%      Default |
018||                               |                      |                  N/A |
019|+-------------------------------+----------------------+----------------------+
020||   3  Tesla V100-SXM2...  On   | 00000000:B2:00.0 Off |                    0 |
021|| N/A   29C    P0    57W / 300W |  15462MiB / 16160MiB |      0%      Default |
022||                               |                      |                  N/A |
023|+-------------------------------+----------------------+----------------------+
024|
025|+-----------------------------------------------------------------------------+
026|| Processes:                                                                  |
027||  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
028||        ID   ID                                                   Usage      |
```
2を入力(2.次へ)
```
/home/ユーザ名/opt/wd_dev/examples/config/02-1grid-1nm-8gpu/work/tmp/q0000-nvidia-smi.txt
1.前へ 2.次へ 3.最後へ 4.左へ 5.右へ 0.戻る
029||=============================================================================|
030||    0   N/A  N/A    210485      C   python                          15459MiB |
031||    1   N/A  N/A    210486      C   python                          15459MiB |
032||    2   N/A  N/A    210497      C   python                          15459MiB |
033||    3   N/A  N/A    210498      C   python                          15459MiB |
034|+-----------------------------------------------------------------------------+
035|
```
正常に動作しているようです。

### 3. 実行結果
```
001|├── gpu
002|│   ├── 000-grid
003|│   │   ├── 0000-q0001-g0-finished
004|│   │   ├── 0001-q0000-g2-finished
005|│   │   ├── 0002-q0001-g3-finished
006|│   │   ├── 0003-q0001-g1-finished
007|│   │   ├── 0004-q0000-g3-finished
008|│   │   ├── 0005-q0000-g3-finished
009|│   │   ├── 0006-q0000-g1-finished
010|│   │   ├── 0007-q0000-g2-finished
011|│   │   ├── 0008-q0000-g0-finished
012|│   │   ├── 0009-q0000-g1-finished
013|│   │   ├── 0010-q0001-g3-finished
014|│   │   ├── 0011-q0001-g2-finished
015|│   │   ├── 0012-q0001-g1-finished
016|│   │   ├── 0013-q0000-g0-finished
017|│   │   ├── 0014-q0000-g2-finished
018|│   │   ├── 0015-q0000-g3-finished
019|│   │   ├── 0016-q0001-g0-finished
020|│   │   ├── 0017-q0000-g0-finished
021|│   │   ├── 0018-q0001-g3-finished
022|│   │   ├── 0019-q0001-g2-finished
023|│   │   ├── 0020-q0001-g0-finished
024|│   │   ├── 0021-q0000-g1-finished
025|│   │   ├── 0022-q0000-g3-finished
026|│   │   └── 0023-q0000-g2-finished
027|│   └── 001-nm
028|│       ├── 0000-q0000-g1-finished
029|│       ├── 0001-q0001-g2-finished
030|│       ├── 0002-q0000-g0-finished
031|│       ├── 0003-q0001-g0-finished
032|│       ├── 0004-q0001-g1-finished
033|│       ├── 0005-q0000-g0-finished
034|│       ├── 0006-q0000-g0-finished
035|│       └── 0007-q0000-g0-finished
```

<!-- ----- ----- ----- --><a id="wdのデバッグ事例"></a>
# wdのデバッグ事例
[目次へ](#目次)

それなりの規模でのwdの使用事例をご紹介したいのですが。
abciのpointを消費してしまう、つまり、大きく課金されてしまうという問題がございます。
そのため、デバッグ用に作成したqsub requestを出したふりをして、子プロセスとして実行する機能を使用して、それなりの規模の実行事例をご紹介いたします。  

なお、インタラクティブノードで実行するには重すぎますので、
計算ノードrt_C.small(執筆時点でspot1時間0.2ポイント)を1つ使用いたします。

また、いわゆるPC linuxに、rt_C.smallとほぼ同じ環境を用意できれば、
そちらのPCで実行することも可能です。

評価関数としては、  
~/opt/wd_dev/examples/config/00-1grid-0gpu/000-grid1/[user.py](examples/config/00-1grid-0gpu/000-grid1/user.py)  
と同じものを使用します。

## 1. qsubを実行したふりをするデバッグモードtest_no_qsubの設定方法

config.ymlでは無く、ソースファイルである  
~/opt/wd_dev/wd/[common.py](wd/common.py)  
を編集します。

編集後、wdはデバッグモードで動作します。
再度common.pyを編集し元に戻すことにより、
通常モードに戻すことができます。

common.pyの
```
# test_*は、テスト用の設定。通常実行時は、全てFalseに。

# False: 通常実行。
# True: qsubは実行せずに、qsubファイルをprocessとして実行。
test_no_qsub = False
```
のtest_no_qsubの部分を、
```
# test_*は、テスト用の設定。通常実行時は、全てFalseに。

# False: 通常実行。
# True: qsubは実行せずに、qsubファイルをprocessとして実行。
test_no_qsub = True
```
と変更します。

## 2. PYTHONPATHを設定するmake_wd_env_for_dev.source

wd_dev/wd以下のソースファイルを変更すると、
再度、インストールが必要になります。  
これはかなりめんどうなので、よろしければ、開発時に使用している  
~/opt/wd_dev/tools/[make_wd_env_for_dev.source](tools/make_wd_env_for_dev.source)  
を、  
~/opt/wd_dev/tools/[make_wd_env.source](tools/make_wd_env.source)  
の変わりにお使いいただければ。

## 3. 1つのaiaccelで多数のtrial

grid search1つ、gpu20個で、try数100回。

~/opt/wd_dev/examples/config/03-1grid-20gpu/000-grid/[config.yml](examples/config/03-1grid-20gpu/000-grid/config.yml)

~/opt/wd_dev/examples/config/03-1grid-20gpu/000-grid/[user.py](examples/config/03-1grid-20gpu/000-grid/user.py)  
C.small(1時間0.2ポイント)とはいえ課金が発生するにもかかわらず申しわけありませんが、
time.sleep(30)
を入れさせていただいています。
辞書順で検索して空いたgpuから使用するため、
評価関数が瞬時に終わると、均等にgpuを使用していることが、分かりづらくなるため。

実行時に忘れやすい点。  
~/opt/wd_dev/wd/[common.py](wd/common.py)を  
test_no_qsub = True  
と変更してqsubを実行したふりをさせる。

実行時のコマンドなど。
```
ssh abci
grep test_no_qsub ~/opt/wd_dev/wd/common.py
qrsh -g ご自身のグループ名 -l rt_C.small=1 -l h_rt=2:00:00
source ~/opt/wd_dev/tools/make_wd_env_for_dev.source
cd ~/opt/wd_dev/examples/config/03-1grid-20gpu
rm -rf work 000-grid/work
python -m wd.bin.cui
```
メニューを選択して実行。

パラメータ。
```
aiaccelの数: 1
num_nodeの合計: 20
trial_numberの合計: 100
計算ノードの合計: 5
G.large: 5
G.small: 0
```

実行すると、全てのaiaccelの出力を表示するため、
終わるまで待ってから、wd.bin.cuiなどで状況をご確認いただきたく。

ご確認後、すぐにexitを忘れずに。
```
exit
```

<details><summary>実行結果(クリックで表示)。計算ノード5つを20回づつ均等に使用。</summary><div>

```
tree ~/opt/wd_dev/examples/config/03-1grid-20gpu/work/gpu/000-grid
├── 0000-q0001-g0-finished
├── 0001-q0003-g1-finished
├── 0002-q0000-g0-finished
├── 0003-q0004-g2-finished
├── 0004-q0000-g2-finished
├── 0005-q0003-g3-finished
├── 0006-q0000-g3-finished
├── 0007-q0003-g0-finished
├── 0008-q0001-g3-finished
├── 0009-q0001-g1-finished
├── 0010-q0004-g0-finished
├── 0011-q0000-g1-finished
├── 0012-q0002-g1-finished
├── 0013-q0003-g2-finished
├── 0014-q0002-g2-finished
├── 0015-q0004-g3-finished
├── 0016-q0004-g1-finished
├── 0017-q0002-g3-finished
├── 0018-q0002-g0-finished
├── 0019-q0001-g2-finished
├── 0020-q0000-g2-finished
├── 0021-q0004-g2-finished
├── 0022-q0002-g0-finished
├── 0023-q0001-g0-finished
├── 0024-q0001-g2-finished
├── 0025-q0001-g1-finished
├── 0026-q0001-g3-finished
├── 0027-q0004-g0-finished
├── 0028-q0004-g1-finished
├── 0029-q0003-g2-finished
├── 0030-q0002-g2-finished
├── 0031-q0000-g0-finished
├── 0032-q0002-g1-finished
├── 0033-q0003-g0-finished
├── 0034-q0002-g3-finished
├── 0035-q0000-g1-finished
├── 0036-q0000-g3-finished
├── 0037-q0003-g1-finished
├── 0038-q0004-g3-finished
├── 0039-q0003-g3-finished
├── 0040-q0001-g2-finished
├── 0041-q0001-g0-finished
├── 0042-q0000-g0-finished
├── 0043-q0000-g3-finished
├── 0044-q0004-g2-finished
├── 0045-q0001-g3-finished
├── 0046-q0001-g1-finished
├── 0047-q0003-g2-finished
├── 0048-q0000-g1-finished
├── 0049-q0000-g2-finished
├── 0050-q0003-g0-finished
├── 0051-q0003-g1-finished
├── 0052-q0003-g3-finished
├── 0053-q0002-g3-finished
├── 0054-q0002-g1-finished
├── 0055-q0002-g2-finished
├── 0056-q0004-g3-finished
├── 0057-q0002-g0-finished
├── 0058-q0004-g0-finished
├── 0059-q0004-g1-finished
├── 0060-q0003-g2-finished
├── 0061-q0001-g0-finished
├── 0062-q0003-g0-finished
├── 0063-q0000-g3-finished
├── 0064-q0000-g1-finished
├── 0065-q0003-g1-finished
├── 0066-q0000-g2-finished
├── 0067-q0001-g1-finished
├── 0068-q0000-g0-finished
├── 0069-q0003-g3-finished
├── 0070-q0001-g2-finished
├── 0071-q0002-g0-finished
├── 0072-q0004-g1-finished
├── 0073-q0004-g0-finished
├── 0074-q0004-g2-finished
├── 0075-q0002-g2-finished
├── 0076-q0001-g3-finished
├── 0077-q0002-g3-finished
├── 0078-q0004-g3-finished
├── 0079-q0002-g1-finished
├── 0080-q0000-g3-finished
├── 0081-q0001-g0-finished
├── 0082-q0000-g2-finished
├── 0083-q0000-g0-finished
├── 0084-q0000-g1-finished
├── 0085-q0002-g1-finished
├── 0086-q0001-g2-finished
├── 0087-q0001-g1-finished
├── 0088-q0003-g0-finished
├── 0089-q0003-g2-finished
├── 0090-q0002-g2-finished
├── 0091-q0004-g2-finished
├── 0092-q0004-g3-finished
├── 0093-q0002-g0-finished
├── 0094-q0004-g1-finished
├── 0095-q0004-g0-finished
├── 0096-q0001-g3-finished
├── 0097-q0003-g1-finished
├── 0098-q0003-g3-finished
└── 0099-q0002-g3-finished
```
</div></details>

## 4. 8つのaiaccel

2パラメータのnelder-mead search8つ、gpu8個で、各try数10で総try数80。

nelder-meadは単発になってしまうために、aiaccelとgpuの数を一致させています。

~/opt/wd_dev/examples/config/04-8nm-8gpu/000-nm/[config.yml](examples/config/04-8nm-8gpu/000-nm/config.yml)

~/opt/wd_dev/examples/config/04-8nm-8gpu/000-nm/[user.py](examples/config/04-8nm-8gpu/000-nm/user.py)  
sleepは入れていません。

残りの7つは、
001-nm, 002-nm, 003-nm, 004-nm, 005-nm, 006-nm, 007-nm。
000からの通番のため。ABCIグループ名は000のみで確認。
今回はtest_no_qsub=TrueのためABCIグループ名は使用しない。  
config.yml, user.pyの内容は、000-nmのものと同じ。

実行時に忘れやすい点。  
~/opt/wd_dev/wd/[common.py](wd/common.py)を  
test_no_qsub = True  
と変更してqsubを実行したふりをさせる。

実行時のコマンドなど。
```
ssh abci
grep test_no_qsub ~/opt/wd_dev/wd/common.py
qrsh -g ご自身のグループ名 -l rt_C.small=1 -l h_rt=2:00:00
source ~/opt/wd_dev/tools/make_wd_env_for_dev.source
cd ~/opt/wd_dev/examples/config/04-8nm-8gpu
rm -rf work *-nm/work
python -m wd.bin.cui
```
メニューを選択して実行。

パラメータ。
```
aiaccelの数: 8
num_nodeの合計: 8
trial_numberの合計: 80
計算ノードの合計: 2
G.large: 2
G.small: 0
```

ご確認後、すぐにexitを忘れずに。
```
exit
```

<details><summary>実行結果(クリックで表示)。計算ノード2つ(gpu8つ)を使用。</summary><div>

```
tree ~/opt/wd_dev/examples/config/04-8nm-8gpu/work/gpu
├── 000-nm
│   ├── 0000-q0001-g1-finished
│   ├── 0001-q0001-g0-finished
│   ├── 0002-q0000-g3-finished
│   ├── 0003-q0000-g1-finished
│   ├── 0004-q0000-g1-finished
│   ├── 0005-q0000-g1-finished
│   ├── 0006-q0000-g1-finished
│   ├── 0007-q0000-g1-finished
│   ├── 0008-q0000-g0-finished
│   └── 0009-q0000-g0-finished
├── 001-nm
│   ├── 0000-q0001-g2-finished
│   ├── 0001-q0001-g1-finished
│   ├── 0002-q0001-g0-finished
│   ├── 0003-q0000-g2-finished
│   ├── 0004-q0000-g2-finished
│   ├── 0005-q0000-g3-finished
│   ├── 0006-q0000-g3-finished
│   ├── 0007-q0000-g1-finished
│   ├── 0008-q0000-g3-finished
│   └── 0009-q0000-g0-finished
├── 002-nm
│   ├── 0000-q0000-g3-finished
│   ├── 0001-q0000-g2-finished
│   ├── 0002-q0001-g3-finished
│   ├── 0003-q0000-g2-finished
│   ├── 0004-q0000-g3-finished
│   ├── 0005-q0000-g1-finished
│   ├── 0006-q0000-g3-finished
│   ├── 0007-q0000-g3-finished
│   ├── 0008-q0000-g0-finished
│   └── 0009-q0000-g0-finished
├── 003-nm
│   ├── 0000-q0000-g2-finished
│   ├── 0001-q0000-g1-finished
│   ├── 0002-q0000-g0-finished
│   ├── 0003-q0000-g0-finished
│   ├── 0004-q0000-g0-finished
│   ├── 0005-q0000-g0-finished
│   ├── 0006-q0000-g0-finished
│   ├── 0007-q0000-g0-finished
│   ├── 0008-q0000-g0-finished
│   └── 0009-q0000-g0-finished
├── 004-nm
│   ├── 0000-q0001-g0-finished
│   ├── 0001-q0000-g3-finished
│   ├── 0002-q0001-g3-finished
│   ├── 0003-q0000-g3-finished
│   ├── 0004-q0001-g1-finished
│   ├── 0005-q0001-g1-finished
│   ├── 0006-q0001-g1-finished
│   ├── 0007-q0001-g1-finished
│   ├── 0008-q0000-g1-finished
│   └── 0009-q0000-g1-finished
├── 005-nm
│   ├── 0000-q0000-g0-finished
│   ├── 0001-q0001-g3-finished
│   ├── 0002-q0001-g2-finished
│   ├── 0003-q0000-g0-finished
│   ├── 0004-q0000-g0-finished
│   ├── 0005-q0000-g2-finished
│   ├── 0006-q0000-g0-finished
│   ├── 0007-q0000-g1-finished
│   ├── 0008-q0000-g1-finished
│   └── 0009-q0000-g1-finished
├── 006-nm
│   ├── 0000-q0000-g3-finished
│   ├── 0001-q0000-g2-finished
│   ├── 0002-q0000-g1-finished
│   ├── 0003-q0000-g1-finished
│   ├── 0004-q0000-g1-finished
│   ├── 0005-q0000-g2-finished
│   ├── 0006-q0000-g2-finished
│   ├── 0007-q0000-g2-finished
│   ├── 0008-q0000-g2-finished
│   └── 0009-q0000-g2-finished
└── 007-nm
    ├── 0000-q0001-g2-finished
    ├── 0001-q0001-g1-finished
    ├── 0002-q0001-g0-finished
    ├── 0003-q0000-g3-finished
    ├── 0004-q0001-g0-finished
    ├── 0005-q0001-g0-finished
    ├── 0006-q0001-g0-finished
    ├── 0007-q0001-g0-finished
    ├── 0008-q0001-g0-finished
    └── 0009-q0001-g0-finished
```
</div></details>

<!-- ----- ----- ----- --><a id="wdのその他のコマンドなど"></a>
# wdのその他のコマンドなど
[目次へ](#目次)

## 1. ndへのプログラムrequest用のpython -m wd.bin.nreq

実際に本機能を使用した
[wdの使用事例-2つのaiaccel](#wdの使用事例-2つのaiaccel)
を元にご説明します。

python -m wd.bin.ndは、計算ノード毎に1つ実行されています。

```
├── node
│   ├── q0000
│   │   ├── run_ai.py-000-running
│   │   ├── run_aiaccel.py-000-running
│   │   ├── run_aiaccel.py-001-running
│   │   └── run_gpu.py-000-running
│   └── q0001
│       └── run_ai.py-000-running
```
本事例では、計算ノードq000とq001で実行されています。

```
python -m wd.bin.nreq 計算ノード名 実行ファイルのパス
```
という書式で、requestします。
実際、前述の
```
python -m wd.bin.cui
の
3を入力(3.コマンド(ps,qstat,qdel-all,nreq-ls,nvidia-smiなど))  
4を入力(4.nreq nvidia-smi.py q0000)
```
の事例では、
```
    def nreq_nvidia_smi(self, qn='q0000'):
        cmd = ('python -m wd.bin.nreq %s '
               '$WD_DEV_DIR/examples/nreq/nvidia-smi.py' % qn)
        self.exec_cmd(cmd)

を実行しています。
wd.bin.nreqは、
指定されたファイルを、指定された計算ノード用のフォルダに、
排他制御のために、まずtempファイルとしてcopyして、その後に、
実行順とrequestをファイル名に付加して、moveします。
前述の事例では、
~/opt/wd_dev/examples/nreq/nvidia-smi.py
が、tempファイルとしてcopy後、
~/opt/wd_dev/examples/config/02-1grid-1nm-8gpu/work/node/q0000/nvidia-smi.py-000-request
として、moveされ、末尾が-requestのファイルを、nd.pyが実行し、
終了したら、末尾の-requestを-finishedに変更します。
~/opt/wd_dev/examples/nreq/nvidia-smi.pyに、

import sys
import os

node_name = sys.argv[1]
nn_wd_path = sys.argv[2]

cmd = ('nvidia-smi >> '
       '%s/work/tmp/%s-nvidia-smi.txt' % (nn_wd_path, node_name))
os.system(cmd)

との記述があるため、
~/opt/wd_dev/examples/config/02-1grid-1nm-8gpu/work/tmp/q0000-nvidia-smi.txt
に、ファイルを作成します。
なお、nreqで実行されるpythonファイルの引数として、
wdのrootパス(今回の場合、~/opt/wd_dev/examples/config/02-1grid-1nm-8gpu)と
計算ノード名が渡されます。
```
<!-- ----- ----- ----- --><a id="wdの試作状況"></a>
# wdの試作状況
[目次へ](#目次)

以上、具体的な使用法を示しながら、wdの試作状況をご説明いたしました。

現在、下記の大幅な変更作業中です。

```
1. 現状の説明
1-1. 排他制御にはディレクトリの一意性を利用
1-2. 各計算ノードがほぼ独立している分散処理

2. 現状の利点
2-1. 単純で作り易い

3. 現状の問題点
3-1. 再開、トラブル処理などが複雑に
3-2. トラブル時などにログが分散しているため見づらい

4. 作成中の次バージョンの解決策
4-1. TCP/IP通信を利用して集中管理に
```
