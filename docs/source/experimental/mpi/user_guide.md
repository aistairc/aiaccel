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
ほぼ全てのフォルダがこちらに作成されますが、
`~/.keras`は、ホームフォルダに作成されます。
次に、mpi環境用のvenv、`~/mpi_work/mpienv`を作成します。
これにより、cpuのみによるテスト実行が可能になります。
さらに、gpuによるテスト実行のためのvenv、`~/mpi_work/tfenv`を作成します。
この部分を入れ替えて、実際に使用されると思われます。

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
module load python/3.11/3.11.2
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
module load python/3.11/3.11.2
module load cuda/11.0/11.0.3
module load cudnn/8.0/8.0.5
cd ~/mpi_work
python3 -m venv tfenv
source tfenv/bin/activate
pip install --upgrade pip setuptools
pip install tensorflow==2.4.1
git clone https://github.com/keras-team/keras-io.git
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
python keras-io/examples/vision/mnist_convnet.py
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
そのため、qrshはせずに、インタラクティブノードで実行しています。
前述のgit cloneによる`~/mpi_work/aiaccel`があることを前提にしています。
```
ssh abci
source /etc/profile.d/modules.sh
module load gcc/11.2.0
module load python/3.8/3.8.13
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
  num_node: 1
  mpi_npernode: 1
  mpi_enviroment: "ABCI"
  mpi_bat_rt_type: "C.small"
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
PYTHONPATH=~/mpi_work/aiaccel python -m aiaccel.experimental.mpi.cli.start --config config.yaml --clean
```
下記のmpi用のフォルダとファイルが、
`~/mpi_work/aiaccel/examples/experimental/mpi/sphere_cpu_1node_1try`に作成されます
(テスト時のものを掲載しています)。
```
hostfile
qsub.sh
qsub.sh.o39165552
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
g0036 slots=1
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

#$ -l rt_C.small=1
#$ -l h_rt=2:00:00
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh
module load gcc/11.2.0
module load python/3.8/3.8.13
module load openmpi/4.1.3
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

### qsub.sh.o39165552

qsub.shの出力ファイルです。abciがカレントフォルダに出力します。
```
--------------------------------------------------------------------------
A process has executed an operation involving a call to the
"fork()" system call to create a child process.  Open MPI is currently
operating in a condition that could result in memory corruption or
other system errors; your job may hang, crash, or produce silent
data corruption.  The use of fork() (or system() or other calls that
create child processes) is strongly discouraged.

The process that invoked fork was:

  Local host:          [[41820,1],0] (PID 92003)

If you are *absolutely sure* that your application will successfully
and correctly survive a call to fork(), you may disable this warning
by setting the mpi_warn_on_fork MCA parameter to 0.
--------------------------------------------------------------------------
Cleaning workspace
Workspace directory /home/*/mpi_work/sphere_cpu_1node_1try/work is cleaned.
config: /home/*/mpi_work/sphere_cpu_1node_1try/config.yaml
Optimizer DEBUG    create numpy random generator by seed: 42
Scheduler DEBUG    create numpy random generator by seed: 42
0,0,g0036.abci.local,"prepare: rank=0 tag=0",20230320-193741-676
Master    INFO     0/1 finished, ready: 0 ,running: 0, end estimated time: Unknown
(中略)
Master    INFO     Master finished.
Optimizer INFO     Optimizer delete alive file.
Optimizer INFO     Optimizer finished.
Scheduler INFO     Scheduler finished.
moving...
Best result    : results/20230320_193749/result/000000.hp
               : 40.07614290457259
Total time [s] : 8
Done.
[g0036.abci.local:91995] 1 more process has sent help message help-opal-runtime.txt / opal_init:warn-fork
[g0036.abci.local:91995] Set MCA parameter "orte_base_help_aggregate" to 0 to see all help / error messages
```

# WIP

### work

aiaccelの出力フォルダです。
`work/experimental/mpi/
```
ls work/experimental/mpi/rank_log/
0.csv  1.csv
cd work/experimental/mpi/rank_log
cat 0.csv
0,0,g0036.abci.local,"prepare: rank=0 tag=0",20230320-193741-676
1,0,g0036.abci.local,"submit start: recv: tag=1 list=[1, 'g0036.abci.local']",20230320-193745-954
cat 1.csv
1,1,g0036.abci.local,"_func_sub(): tag=1 command=['python', 'user.py', '2>', '/home/*/mpi_work/sphere_cpu_1node_1try/work/error/0.txt', '--trial_id', '0', '--config', '/home/*/mpi_work/sphere_cpu_1node_1try/config.yaml', '--x1=-1.254598811526375', '--x2=4.507143064099161', '--x3=2.3199394181140507', '--x4=0.986584841970366', '--x5=-3.439813595575635']",20230320-193745-951
1,1,g0036.abci.local,"_func_sub(): debug: line=objective_y:40.07614290457259",20230320-193747-405
1,1,g0036.abci.local,"_func_sub(): end: tag=1 process finished.",20230320-193747-494
```


### gpuでの使用方法

```
ssh abci
source /etc/profile.d/modules.sh
module load gcc/11.2.0
module load python/3.8/3.8.13
source ~/mpienv/bin/activate
export PYTHONPATH=~/aiaccel
cd ~/aiaccel/examples/experimental/mpi/mnist_convnet
python -m aiaccel.experimental.mpi.cli.start --config config.yaml --clean
```

# その他の機能、情報など

## qsub先でのnvidia-smiなどのコマンドの実行方法
```
python -m aiaccel.experimental.mpi.cli.qrsh-inherit 0 'nvidia-smi'
python -m aiaccel.experimental.mpi.cli.qrsh-inherit 0 'ps -ef | grep acb'
```
