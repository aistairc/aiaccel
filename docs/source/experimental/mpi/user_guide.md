# mpi機能の概要

abciでmpiによりaiaccelのjobを並列実行させます。
mpi機能の実行環境は、通常のaiaccelとは完全に分離されています。
そのため、aiaccelインストール後に、mpi環境を整えて、mpi用aiaccelを実行します。

なお、後述の使い方などの説明では、
```
~/mpi_work
```
のフォルダを作成し、その構造を前提とします
(config.yamlの設定などで構造を変更することは可能です)。

# インストール

まず、`~/mpi_work`フォルダを作成します。
```
mkdir ~/mpi_work
```
次に、mpi環境用のvenv、`~/mpi_work/mpienv`を作成します。
これにより、cpuのみによるテスト実行が可能になります。
さらに、gpuによるテスト実行のためのvenv、`~/mpi_work/ptenv`を作成します。
この部分などを入れ替えて、実際に使用されると思われます。

## `~/mpi_work/mpienv`を作成(aiaccelとmpi4pyをインストール)

通常のaiaccelのインスールとほぼ同じですが、
```
module load hpcx-mt/2.12
pip install mpi4py
```
などの部分が追加されています。

また、main branchへのmerge前のため、通常、
```
pip install git+https://github.com/aistairc/aiaccel.git
```
の部分が、
```
pip install git+https://github.com/aistairc/aiaccel.git@feature/experimental/mpi
```
と変更されています。

テスト時の手順を以下に示します。
```
ssh abci
qrsh -g your_group_id -l rt_C.small=1 -l h_rt=2:00:00
source /etc/profile.d/modules.sh
module load python/3.11
module load hpcx-mt/2.12
cd ~/mpi_work
python3 -m venv mpienv
source mpienv/bin/activate
pip install --upgrade pip
pip install git+https://github.com/aistairc/aiaccel.git@feature/experimental/mpi
pip install mpi4py
deactivate
exit
```

## `~/mpi_work/ptenv`を作成した後にmnist.pyの動作確認

まず、gpuでのテストのためのvenv、`~/mpi_work/ptenv`を作成します。
その後に、gpuでのテストに使用する評価関数の動作確認をします。
```
ssh abci
qrsh -g your_group_id -l rt_G.small=1 -l h_rt=2:00:00
source /etc/profile.d/modules.sh
module load python/3.11
module load cuda/11.8
module load cudnn/8.6
cd ~/mpi_work
python3 -m venv ptenv
source ptenv/bin/activate
pip install --upgrade pip setuptools
pip install torch torchvision torchaudio
git clone https://github.com/pytorch/examples.git
cd examples/mnist
python3 main.py
deactivate
exit
```

# 使用方法

## mpi機能付きのaiaccelをgit clone

まず、mpi機能付きのaiaccelをgit cloneします。
main branchへのmerge前のため、branch `feature/experimental/mpi`を指定しています。
```
ssh abci
cd ~/mpi_work
git clone -b feature/experimental/mpi https://github.com/aistairc/aiaccel.git
```

## cpuでの使用方法

cpuとgpuで、パラメータの設定が若干違います。
まずは、cpuでの使用方法を説明します。

mpiによるqsubでの実行の指示のみのため、負荷がほとんどかかりません。
そのため、qrshはせずに、インタラクティブノードで実行します。
前述の`~/mpi_work/aiaccel`などがあることを前提にしています。
```
ssh abci
source /etc/profile.d/modules.sh
module load python/3.11
module load hpcx-mt/2.12
cd ~/mpi_work
source mpienv/bin/activate
cp -r aiaccel/examples/experimental/mpi/sphere_cpu_1node_1try .
cd sphere_cpu_1node_1try
```
vi config.yamlなどで、
```
  group: "[your_group_id]"
```
の`your_group_id`を、適切なグループidに変更して下さい。

config.yamlのmpi関連のパラメータを以下に抜粋します。
```
generic:
  workspace: "./work"

resource:
  type: "mpi"
  num_workers: 1
  mpi_npernode: 40
  mpi_bat_rt_num: 1
  mpi_bat_h_rt: "2:00:00"
  mpi_bat_root_dir: "~/mpi_work"
  mpi_bat_venv_dir: "./mpienv"
  mpi_bat_aiaccel_dir: "./aiaccel"
  mpi_bat_config_dir: "./sphere_cpu_1node_1try"
  mpi_bat_file: "./qsub.sh"
  mpi_hostfile: "./hostfile"
  mpi_gpu_mode: False
  mpi_bat_make_file: True

ABCI:
  group: "[your_group_id]"

optimize:
  trial_number: 1
```
詳細は後述しますが、
1tryだけでmpiの動作テストを行うため、
qsubのリクエスト待ち後の実行開始後、1分程で終了します。

```
PYTHONPATH=~/mpi_work/aiaccel python -m aiaccel.experimental.mpi.cli.start --config config.yaml --clean &
```
abciの混雑状況にもよりますが、しばらくすると、
```
Your job 39893216 ("qsub.sh") has been submitted
 < qsub -g your_group_id ./qsub.sh
```
とのqsub requestが受け付けられた旨のメッセージが表示されます(テスト時のものを掲載しています)。

さらに1分程待つと、下記のフォルダとファイルが、
`~/mpi_work/aiaccel/examples/experimental/mpi/sphere_cpu_1node_1try`に作成されます
(テスト時のものを掲載しています)。
(config.yaml)と(user.py)は実行前から存在している設定ファイルなどです。
```
(config.yaml)
hostfile
qsub.sh
qsub.sh.o39893256
results/
(user.py)
work/
```
個別に説明します。

### hostfile

通常、hostfileをユーザが閲覧、編集することはありません。
トラブル時に内容を確認できるように指定しています。
config.yamlの`mpi_hostfile: ./hostfile`で指定されたファイル名です。
mpi実行時に作成されます。
なお、mpi_hostfileは、`mpi_bat_config_dir`からの相対指定です。
以下、config.yamlの設定に従いテスト時に作成されたhostfileの内容です。
```
g0170 slots=40
```

### qsub.sh

config.yamlの`mpi_bat_file: ./qsub.sh`で指定されたファイル名です。
`mpi_bat_make_file: True`であれば、mpi実行時に作成されます。
Falseの時は作成されません。
なお、mpi_bat_fileは、`mpi_bat_config_dir`からの相対指定です。
以下、config.yamlの設定に従いテスト時に作成されたqsub.shの内容です
(mpiexecからの3行は元々1行だったものを見易いように加工しています)。
```
#!/bin/bash

#$ -l rt_F=1
#$ -l h_rt=2:00:00
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.11
module load hpcx-mt/2.12
source ~/mpi_work/mpienv/bin/activate
export PYTHONPATH=~/mpi_work/aiaccel/:$PYTHONPATH

cd ~/mpi_work/sphere_cpu_1node_1try

python -m aiaccel.experimental.mpi.cli.start --config config.yaml --make_hostfile

mpiexec -n 2 -hostfile ~/mpi_work/sphere_cpu_1node_1try/hostfile \
python -m mpi4py.futures -m aiaccel.experimental.mpi.cli.start \
--config config.yaml --clean --from_mpi_bat

deactivate
```
qsub.shを直接変更したい場合は、config.yamlで`mpi_bat_make_file: False`とします。

### qsub.sh.o39893256

qsub.shの出力ファイルです。abciがqsub request時のカレントフォルダに出力します。
```
Cleaning workspace
Workspace directory /home/acb11523fz/mpi_work/sphere_cpu_1node_1try/work is cleaned.
config: /home/acb11523fz/mpi_work/sphere_cpu_1node_1try/config.yaml
Optimizer DEBUG    create numpy random generator by seed: 42
Scheduler DEBUG    create numpy random generator by seed: 42
0,0,g0170.abci.local,"prepare: rank=0 tag=0",20230529-104027-515
Master    INFO     0/1 finished, ready: 0 ,running: 0, end estimated time: Unknown
Optimizer INFO     hp_ready: 0, hp_running: 0, hp_finished: 0, total: 1, pool_size: 1
Scheduler DEBUG    Submit a job: 0
(中略)
Master    INFO     Best hyperparameter is followings:
Master    INFO     [{'trial_id': 0, 'parameters': [{'parameter_name': 'x1', 'type': 'FLOAT', 'value': -1.254598811526375}, {'parameter_name': 'x2', 'type': 'FLOAT', 'value': 4.507143064099161}, {'parameter_name': 'x3', 'type': 'FLOAT', 'value': 2.3199394181140507}, {'parameter_name': 'x4', 'type': 'FLOAT', 'value': 0.986584841970366}, {'parameter_name': 'x5', 'type': 'FLOAT', 'value': -3.439813595575635}], 'result': [40.07614290457259], 'start_time': '05/29/2023 10:40:29', 'end_time': '05/29/2023 10:40:30'}]
Master    INFO     Master finished.
Optimizer INFO     Optimizer delete alive file.
Optimizer INFO     Optimizer finished.
Scheduler INFO     Scheduler finished.
moving...
Best result [0] : results/20230529_104032/result/0.hp
        value : 40.07614290457259
Total time [s] : 5
Done.
```

### results/

aiaccelの結果の出力フォルダです。
詳細についてはaiaccel本体のマニュアルなどをご参照いただきたく。

### work/

aiaccelの作業時の出力フォルダです。
全般的な詳細についてはaiaccel本体のマニュアルなどをご参照いただきたく。

ここでは、mpiが出力するフォルダについて説明いたします。

#### work/experimental/mpi/rank_log/

mpiのdebug用のフォルダです。
各rankの動作状況が出力されています。
n.csvのnがmpiのrank番号に対応しています。

```
ls work/experimental/mpi/rank_log/
0.csv  1.csv

cd work/experimental/mpi/rank_log

cat 0.csv
0,0,g0170.abci.local,"prepare: rank=0 tag=0",20230529-104027-515
1,0,g0170.abci.local,"submit start: recv: tag=1 trial_id=0 list=[1, 'g0170.abci.local']",20230529-104029-314

cat 1.csv
1,1,g0170.abci.local,"_func_sub(): tag=1 command=['python', 'user.py', '--x1', '-1.254598811526375', '--x2', '4.507143064099161', '--x3', '2.3199394181140507', '--x4', '0.986584841970366', '--x5', '-3.439813595575635', '--trial_id', '0', '--config', '/home/acb11523fz/mpi_work/sphere_cpu_1node_1try/config.yaml', '2>', '/home/acb11523fz/mpi_work/sphere_cpu_1node_1try/work/error/0.txt']",20230529-104029-313
1,1,g0170.abci.local,"_func_sub(): debug: line=40.07614290457259",20230529-104030-576
1,1,g0170.abci.local,"_func_sub(): end: tag=1 process finished.",20230529-104030-728
```

## gpuでの使用方法

gpuでの使用方法を説明します。

mpiによるqsubでの実行の指示のみのため、負荷がほとんどかかりません。
そのため、qrshはせずに、インタラクティブノードで実行します。
前述の`~/mpi_work/aiaccel`などがあることを前提にしています。
```
ssh abci
source /etc/profile.d/modules.sh
module load python/3.11
module load hpcx-mt/2.12
cd ~/mpi_work
source mpienv/bin/activate
cp -r aiaccel/examples/experimental/mpi/mnist_gpu_2node_8try .
cd mnist_gpu_2node_8try
```
`main.py`は
`~/mpi_work/examples/mnist/main.py`
を修正したものですが、
dataをdownloadしないように変更してあります。
そのため、data/をmvしておきます。
```
mv ~/mpi_work/examples/data ~/mpi_work
```
また、`mnist.sh`に実行権限を付与しておきます。
```
chmod u+x ~/mpi_work/mnist_gpu_2node_8try/mnist.sh
```
vi config.yamlなどで、
```
  group: "[your_group_id]"
```
の`your_group_id`を、適切なグループidに変更して下さい。

config.yamlのmpi関連のパラメータを以下に抜粋します。
```
generic:
  workspace: "./work"

resource:
  type: "mpi"
  num_workers: 8
  mpi_npernode: 4
  mpi_bat_rt_num: 2
  mpi_bat_h_rt: "2:00:00"
  mpi_bat_root_dir: "~/mpi_work"
  mpi_bat_venv_dir: "./mpienv"
  mpi_bat_aiaccel_dir: "./aiaccel"
  mpi_bat_config_dir: "./sphere_cpu_1node_1try"
  mpi_bat_file: "./qsub.sh"
  mpi_hostfile: "./hostfile"
  mpi_gpu_mode: True
  mpi_bat_make_file: True

ABCI:
  group: "[your_group_id]"

optimize:
  trial_number: 8
```
詳細は後述しますが、
2node(1nodeのgpuボードが4枚)8tryでmpiの動作テストを行うため、
qsubのリクエスト待ち後の実行開始後、8並列実行のため、5分程で終了します。

```
PYTHONPATH=~/mpi_work/aiaccel python -m aiaccel.experimental.mpi.cli.start --config config.yaml --clean &
```
abciの混雑状況にもよりますが、しばらくすると、
```
Your job 39894794 ("qsub.sh") has been submitted
 < qsub -g your_group_id ./qsub.sh
```
とのqsub requestが受け付けられた旨のメッセージが表示されます(テスト時のものを掲載しています)。

さらに5分程待つと、下記のフォルダとファイルが、
`~/mpi_work/aiaccel/examples/experimental/mpi/mnist_gpu_2node_8try`に作成されます
(テスト時のものを掲載しています)。
(config.yaml)、(main.py)、(mnist.sh)、(user.py)は実行前から存在している設定ファイルなどです。
```
(config.yaml)
hostfile
(main.py)
(mnist.sh)
qsub.sh
qsub.sh.o39894794
results/
(user.py)
work/
```
個別に説明します。

### hostfile

通常、hostfileをユーザが閲覧、編集することはありません。
トラブル時に内容を確認できるように指定しています。
config.yamlの`mpi_hostfile: ./hostfile`で指定されたファイル名です。
mpi実行時に作成されます。
なお、mpi_hostfileは、`mpi_bat_config_dir`からの相対指定です。
以下、config.yamlの設定に従いテスト時に作成されたhostfileの内容です。
```
g0206 slots=5
g0209 slots=4
```

### qsub.sh

config.yamlの`mpi_bat_file: ./qsub.sh`で指定されたファイル名です。
`mpi_bat_make_file: True`であれば、mpi実行時に作成されます。
Falseの時は作成されません。
なお、mpi_bat_fileは、`mpi_bat_config_dir`からの相対指定です。
以下、config.yamlの設定に従いテスト時に作成されたqsub.shの内容です
(mpiexecからの3行は元々1行だったものを見易いように加工しています)。
```
#!/bin/bash

#$ -l rt_F=2
#$ -l h_rt=2:00:00
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.11
module load hpcx-mt/2.12
source ~/mpi_work/mpienv/bin/activate
export PYTHONPATH=~/mpi_work/aiaccel/:$PYTHONPATH

cd ~/mpi_work/mnist_gpu_2node_8try

python -m aiaccel.experimental.mpi.cli.start --config config.yaml --make_hostfile

mpiexec -n 9 -hostfile ~/mpi_work/mnist_gpu_2node_8try/hostfile \
python -m mpi4py.futures -m aiaccel.experimental.mpi.cli.start \
--config config.yaml --clean --from_mpi_bat

deactivate
```
qsub.shを直接変更したい場合は、config.yamlで`mpi_bat_make_file: False`とします。

### qsub.sh.o39894794

qsub.shの出力ファイルです。abciがqsub request時のカレントフォルダに出力します。
```
Cleaning workspace
Workspace directory /home/acb11523fz/mpi_work/mnist_gpu_2node_8try/work is cleaned.
config: /home/acb11523fz/mpi_work/mnist_gpu_2node_8try/config.yaml
Optimizer DEBUG    create numpy random generator by seed: 42
Scheduler DEBUG    create numpy random generator by seed: 42
0,0,g0206.abci.local,"prepare: rank=0 tag=0",20230529-173849-299
Master    INFO     0/8 finished, ready: 0 ,running: 0, end estimated time: Unknown
Optimizer INFO     hp_ready: 0, hp_running: 0, hp_finished: 0, total: 8, pool_size: 8
Scheduler DEBUG    Submit a job: 0
(中略)
Scheduler INFO     8/8, finished, ready: 0, running: 0
Master    INFO     Best hyperparameter is followings:
Master    INFO     [{'trial_id': 3, 'parameters': [{'parameter_name': 'x1', 'type': 'FLOAT', 'value': 1.177585271554666}], 'result': [0.024746950340270996], 'start_time': '05/29/2023 17:38:54', 'end_time': '05/29/2023 17:41:52'}]
Master    INFO     Master finished.
Optimizer INFO     Optimizer delete alive file.
Optimizer INFO     Optimizer finished.
Scheduler INFO     Scheduler finished.
moving...
Best result [0] : results/20230529_174221/result/3.hp
        value : 0.024746950340270996
Total time [s] : 212
Done.
```

### results/

aiaccelの結果の出力フォルダです。
詳細についてはaiaccel本体のマニュアルなどをご参照いただきたく。

### work/

aiaccelの作業時の出力フォルダです。
全般的な詳細についてはaiaccel本体のマニュアルなどをご参照いただきたく。

ここでは、mpiが出力するフォルダについて説明いたします。

#### work/experimental/mpi/rank_log/

mpiのdebug用のフォルダです。
各rankの動作状況が出力されています。
n.csvのnがmpiのrank番号に対応しています。

```
ls work/experimental/mpi/rank_log/
0.csv  1.csv  2.csv  3.csv  4.csv  5.csv  6.csv  7.csv  8.csv

cd work/experimental/mpi/rank_log

cat 0.csv
0,0,g0206.abci.local,"prepare: rank=0 tag=0",20230529-173849-299
1,0,g0206.abci.local,"submit start: recv: tag=1 trial_id=0 list=[1, 'g0206.abci.local']",20230529-173852-990
1,0,g0206.abci.local,"send: rank=1 tag=1 gpu_num=0",20230529-173852-991
1,0,g0206.abci.local,"info: gpu_list={'g0206.abci.local': [[0, 1]]}",20230529-173852-991
2,0,g0206.abci.local,"submit start: recv: tag=2 trial_id=1 list=[2, 'g0206.abci.local']",20230529-173853-319
2,0,g0206.abci.local,"send: rank=2 tag=2 gpu_num=1",20230529-173853-319
2,0,g0206.abci.local,"info: gpu_list={'g0206.abci.local': [[0, 1], [1, 2]]}",20230529-173853-320
3,0,g0206.abci.local,"submit start: recv: tag=3 trial_id=2 list=[3, 'g0206.abci.local']",20230529-173853-616
3,0,g0206.abci.local,"send: rank=3 tag=3 gpu_num=2",20230529-173853-664
3,0,g0206.abci.local,"info: gpu_list={'g0206.abci.local': [[0, 1], [1, 2], [2, 3]]}",20230529-173853-675
4,0,g0206.abci.local,"submit start: recv: tag=4 trial_id=3 list=[4, 'g0206.abci.local']",20230529-173853-965
4,0,g0206.abci.local,"send: rank=4 tag=4 gpu_num=3",20230529-173854-036
4,0,g0206.abci.local,"info: gpu_list={'g0206.abci.local': [[0, 1], [1, 2], [2, 3], [3, 4]]}",20230529-173854-036
5,0,g0206.abci.local,"submit start: recv: tag=5 trial_id=4 list=[5, 'g0209.abci.local']",20230529-173854-312
5,0,g0206.abci.local,"send: rank=5 tag=5 gpu_num=0",20230529-173854-356
5,0,g0206.abci.local,"info: gpu_list={'g0206.abci.local': [[0, 1], [1, 2], [2, 3], [3, 4]], 'g0209.abci.local': [[0, 5]]}",20230529-173854-356
6,0,g0206.abci.local,"submit start: recv: tag=6 trial_id=5 list=[6, 'g0209.abci.local']",20230529-173854-690
6,0,g0206.abci.local,"send: rank=6 tag=6 gpu_num=1",20230529-173854-690
6,0,g0206.abci.local,"info: gpu_list={'g0206.abci.local': [[0, 1], [1, 2], [2, 3], [3, 4]], 'g0209.abci.local': [[0, 5], [1, 6]]}",20230529-173854-691
7,0,g0206.abci.local,"submit start: recv: tag=7 trial_id=6 list=[7, 'g0209.abci.local']",20230529-173855-079
7,0,g0206.abci.local,"send: rank=7 tag=7 gpu_num=2",20230529-173855-091
7,0,g0206.abci.local,"info: gpu_list={'g0206.abci.local': [[0, 1], [1, 2], [2, 3], [3, 4]], 'g0209.abci.local': [[0, 5], [1, 6], [2, 7]]}",20230529-173855-091
8,0,g0206.abci.local,"submit start: recv: tag=8 trial_id=7 list=[8, 'g0209.abci.local']",20230529-173855-421
8,0,g0206.abci.local,"send: rank=8 tag=8 gpu_num=3",20230529-173855-511
8,0,g0206.abci.local,"info: gpu_list={'g0206.abci.local': [[0, 1], [1, 2], [2, 3], [3, 4]], 'g0209.abci.local': [[0, 5], [1, 6], [2, 7], [3, 8]]}",20230529-173855-520

cat 1.csv
1,1,g0206.abci.local,"_func_sub(): tag=1 command=['python', 'user.py', '--x1', '0.7741722139252524', '--trial_id', '0', '--config', '/home/acb11523fz/mpi_work/mnist_gpu_2node_8try/config.yaml', '2>', '/home/acb11523fz/mpi_work/mnist_gpu_2node_8try/work/error/0.txt']",20230529-173852-988
1,1,g0206.abci.local,"start: recv: gpu_num=0",20230529-173852-991
1,1,g0206.abci.local,"_func_sub(): debug: line=Train Epoch: 1 [0/60000 (0%)]    Loss: 2.299825",20230529-173929-970
1,1,g0206.abci.local,"_func_sub(): debug: line=Train Epoch: 1 [640/60000 (1%)]  Loss: 1.391942",20230529-173929-971
1,1,g0206.abci.local,"_func_sub(): debug: line=Train Epoch: 1 [1280/60000 (2%)] Loss: 0.876705",20230529-173929-971
(中略)
1,1,g0206.abci.local,"_func_sub(): debug: line=Train Epoch: 14 [59520/60000 (99%)]      Loss: 0.003401",20230529-174218-899
1,1,g0206.abci.local,"_func_sub(): debug: line=Test set: Average loss: 0.0259, Accuracy: 9915/10000 (99%)",20230529-174218-899
1,1,g0206.abci.local,"_func_sub(): debug: line=0.025880657482147216",20230529-174218-900
1,1,g0206.abci.local,"_func_sub(): debug: line=before break: s=|| save_line=|0.025880657482147216|",20230529-174218-900
1,1,g0206.abci.local,"_func_sub(): debug: line=end: save_line=|0.025880657482147216| ret_s=|0.025880657482147216|",20230529-174218-901
1,1,g0206.abci.local,"_func_sub(): debug: line=0.025880657482147216",20230529-174218-901
1,1,g0206.abci.local,"_func_sub(): end: tag=1 process finished.",20230529-174219-033
```

# WIP

続きは作成中。
