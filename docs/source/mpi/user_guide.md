# mpi 機能の使用方法

`aiaccel` で `mpi` を使用し、jobを並列実行させる手順を以下に示します。<br>
`mpi`を使用する場合は `venv` の作成を推奨します。本手順では `venv` を使用した場合の手順を示します。

# 事前準備


* `mpi`環境のrootフォルダを作成

  今後、mpi環境のrootフォルダとして `~/mpi_work` を使用します。ただし、任意のフォルダを使用可能です。

  ~~~ bash
  mkdir ~/mpi_work
  ~~~

* venv の作成
  ~~~ bash
  cd ~/mpi_work
  python3 -m venv mpienv
  ~~~

# インストール

`mpi` 機能を使用するには、`mpi4py` をインストールする必要があります。

### ローカル環境の場合

~~~ bash
sudo apt-get update
sudo apt-get install python3-dev openmpi-bin libopenmpi-dev libmpich-dev libhwloc-dev libevent-dev
pip install --upgrade pip
pip install mpi4py
pip install git+https://github.com/aistairc/aiaccel.git
~~~


### ABCIの場合

~~~ bash
qrsh -g your_group_id -l rt_C.small=1 -l h_rt=2:00:00
source /etc/profile.d/modules.sh
module load python/3.11
module load hpcx-mt/2.12
cd ~/mpi_work
source mpienv/bin/activate
pip install --upgrade pip
pip install mpi4py
pip install git+https://github.com/aistairc/aiaccel.git
deactivate
exit
~~~


# 使用方法
​
## コンフィグファイルを作成
`mpi` を使用する場合は、コンフィグファイルの `resource.type` を `mpi` に設定します。
また、`resource` には、以下のパラメータを設定します。
- `mpi_npernode` - ノード数を指定します。
- `mpi_enviroment` - mpi環境のvenvのパスを指定します。
- `mpi_bat_rt_type` - qsubのリクエストのタイプを指定します。
- `mpi_bat_rt_num` - qsubのリクエストの数を指定します。
- `mpi_bat_h_rt` - qsubのリクエストの時間を指定します。
- `mpi_bat_root_dir` - mpi環境のrootフォルダのパスを指定します。
- `mpi_bat_venv_dir` - mpi環境のvenvのパスを指定します。ただし、mpi_bat_root_dirからの相対指定です。
- `mpi_bat_aiaccel_dir` - aiaccelのパスを指定します。ただし、mpi_bat_root_dirからの相対指定です。
- `mpi_bat_config_dir` - config.yamlのパスを指定します。ただし、mpi_bat_root_dirからの相対指定です。
- `mpi_bat_file` - qsubのバッチファイルのパスを指定します。ただし、mpi_bat_config_dirからの相対指定です。
- `mpi_hostfile` - mpiのhostfileのパスを指定します。ただし、mpi_bat_config_dirからの相対指定です。
- `mpi_gpu_mode` - gpuを使用する場合はTrue、cpuのみの場合はFalseを指定します。
- `mpi_bat_make_file` - qsubのバッチファイルを作成する場合はTrue、作成しない場合はFalseを指定します。

### 例

~~~ yaml
generic:
    ...
resource:
    type: local
    num_workers: 1
    mpi_npernode: 4
    mpi_enviroment: "ABCI"
    mpi_bat_rt_type: "F"
    mpi_bat_rt_num: 1
    mpi_bat_h_rt: "72:00:00"
    mpi_bat_root_dir: "~/mpi_work"
    mpi_bat_venv_dir: "./mpienv"
    mpi_bat_aiaccel_dir: "./aiaccel"
    mpi_bat_config_dir: "./sphere_cpu_1node_1try"
    mpi_bat_file: "./qsub.sh"
    mpi_hostfile: "./hostfile"
    mpi_gpu_mode: True
    mpi_bat_make_file: True
ABCI:
    ...
optimize:
    ...
job_setting:
    ...
logger:
    ...
~~~


## 実行

実行は次のコマンドで行います。
~~~ bash
aiaccel-start --config config.yaml --clean
~~~

実行後、hostfileとqsub.shを生成します。

- hostfile

  ​hostfileとは、MPIプロセスを実行するノードとプロセス数を指定するファイルです。
  hostfileは`mpi_hostfile`で指定したパスに作成されます。
  通常、hostfileをユーザが閲覧、編集することはありません。

- hostfile の例
  ~~~
  g0047 slots=2
  ~~~

- qsub.sh

  config.yamlの `mpi_bat_file: ./qsub.sh` で指定されたバッチファイルです。
  バッチファイルは、 `mpi_bat_file` で指定したパスに作成します。
   `mpi_bat_make_file: True` であれば、実行時に作成します。
  Falseの場合は作成しません。


実行後、しばらくすると次のようなメッセージを表示します。
~~~ {text}
Your job 40320948 ("qsub.sh") has been submitted
 < qsub -g your_group_id ./qsub.sh
~~~


## 結果の確認

結果は、`qsub.sh.o40320948`に出力します。`o40320948`の部分は、実行時に表示されたメッセージに合わせて変更してください。

### qsub.sh.o40320948

~~~ bash
cat qsub.sh.o40320948
~~~
~~~ {text}
Cleaning workspace
Workspace directory /home/your_user_name/mpi_work/sphere_cpu_1node_1try/work is cleaned.
config: /home/your_user_name/mpi_work/sphere_cpu_1node_1try/config.yaml
^[[34mOptimizer DEBUG    create numpy random generator by seed: 42
^[[0m^[[34mScheduler DEBUG    create numpy random generator by seed: 42
^[[0m0,0,g0047.abci.local,"prepare: rank=0 tag=0",20230731-065737-671
^[[0m^[[0m^[[32mMaster    INFO     0/1 finished, ready: 0 ,running: 0, end estimated time: Unknown
^[[0m^[[32mOptimizer INFO     hp_ready: 0, hp_running: 0, hp_finished: 0, total: 1, pool_size: 1
(中略)
^[[0m^[[32mScheduler INFO     1/1, finished, ready: 0, running: 0
^[[0m^[[32mMaster    INFO     Best hyperparameter is followings:
^[[0m^[[32mMaster    INFO     [{'trial_id': 0, 'parameters': [{'parameter_name': 'x1', 'type': 'uniform_float', 'value': -1.254598811526375}, {'parameter_name': 'x2', 'type': 'uniform_float', 'value': 4.507143064099161}, {'parameter_name': 'x3', 'type': 'uniform_float', 'value': 2.3199394181140507}, {'parameter_name': 'x4', 'type': 'uniform_float', 'value': 0.986584841970366}, {'parameter_name': 'x5', 'type': 'uniform_float', 'value': -3.439813595575635}], 'result': [40.07614290457259], 'start_time': '07/31/2023 06:57:39', 'end_time': '07/31/2023 06:57:41'}]
^[[0m^[[32mMaster    INFO     Master finished.
^[[0m^[[32mOptimizer INFO     Optimizer delete alive file.
^[[0m^[[32mOptimizer INFO     Optimizer finished.
^[[0m^[[32mScheduler INFO     Scheduler finished.
moving...
Best result [0] : results/20230731_065743/result/0.hp
        value : 40.07614290457259
Total time [s] : 5
Done.
^[[0m^[[0m^[[0m
~~~
​

また、mpiのログは、`work//mpi/rank_log/`に出力します。

~~~ bash
ls work/mpi/rank_log/
0.csv  1.csv  2.csv  3.csv  4.csv  5.csv  6.csv  7.csv  8.csv
​
cd work/mpi/rank_log
​
cat 0.csv
~~~

~~~ {text}
0,0,g0006.abci.local,"prepare: rank=0 tag=0",20230731-072329-707
1,0,g0006.abci.local,"submit start: recv: tag=1 trial_id=0 list=[1, 'g0006.abci.local']",20230731-072335-487
1,0,g0006.abci.local,"send: rank=1 tag=1 gpu_num=0",20230731-072335-487
1,0,g0006.abci.local,"info: gpu_list={'g0006.abci.local': [[0, 1]]}",20230731-072335-488
2,0,g0006.abci.local,"submit start: recv: tag=2 trial_id=1 list=[2, 'g0006.abci.local']",20230731-072335-817
2,0,g0006.abci.local,"send: rank=2 tag=2 gpu_num=1",20230731-072335-817
2,0,g0006.abci.local,"info: gpu_list={'g0006.abci.local': [[0, 1], [1, 2]]}",20230731-072335-817
3,0,g0006.abci.local,"submit start: recv: tag=3 trial_id=2 list=[3, 'g0006.abci.local']",20230731-072336-042
3,0,g0006.abci.local,"send: rank=3 tag=3 gpu_num=2",20230731-072336-042
3,0,g0006.abci.local,"info: gpu_list={'g0006.abci.local': [[0, 1], [1, 2], [2, 3]]}",20230731-072336-043
4,0,g0006.abci.local,"submit start: recv: tag=4 trial_id=3 list=[4, 'g0006.abci.local']",20230731-072336-234
4,0,g0006.abci.local,"send: rank=4 tag=4 gpu_num=3",20230731-072336-234
4,0,g0006.abci.local,"info: gpu_list={'g0006.abci.local': [[0, 1], [1, 2], [2, 3], [3, 4]]}",20230731-072336-235
5,0,g0006.abci.local,"submit start: recv: tag=5 trial_id=4 list=[5, 'g0018.abci.local']",20230731-072336-425
5,0,g0006.abci.local,"send: rank=5 tag=5 gpu_num=0",20230731-072336-426
5,0,g0006.abci.local,"info: gpu_list={'g0006.abci.local': [[0, 1], [1, 2], [2, 3], [3, 4]], 'g0018.abci.local': [[0, 5]]}",20230731-072336-426
6,0,g0006.abci.local,"submit start: recv: tag=6 trial_id=5 list=[6, 'g0018.abci.local']",20230731-072336-611
6,0,g0006.abci.local,"send: rank=6 tag=6 gpu_num=1",20230731-072336-612
6,0,g0006.abci.local,"info: gpu_list={'g0006.abci.local': [[0, 1], [1, 2], [2, 3], [3, 4]], 'g0018.abci.local': [[0, 5], [1, 6]]}",20230731-072336-612
7,0,g0006.abci.local,"submit start: recv: tag=7 trial_id=6 list=[7, 'g0018.abci.local']",20230731-072336-733
7,0,g0006.abci.local,"send: rank=7 tag=7 gpu_num=2",20230731-072336-733
7,0,g0006.abci.local,"info: gpu_list={'g0006.abci.local': [[0, 1], [1, 2], [2, 3], [3, 4]], 'g0018.abci.local': [[0, 5], [1, 6], [2, 7]]}",20230731-072336-734
8,0,g0006.abci.local,"submit start: recv: tag=8 trial_id=7 list=[8, 'g0018.abci.local']",20230731-072336-864
8,0,g0006.abci.local,"send: rank=8 tag=8 gpu_num=3",20230731-072336-864
8,0,g0006.abci.local,"info: gpu_list={'g0006.abci.local': [[0, 1], [1, 2], [2, 3], [3, 4]], 'g0018.abci.local': [[0, 5], [1, 6], [2, 7], [3, 8]]}",20230731-072336-865
​
cat 1.csv
1,1,g0006.abci.local,"_func_sub(): tag=1 command=['python', 'user.py', '--x1=0.7741722139252524', '--trial_id=0', '--config=/home/your_user_name/mpi_work/mnist_gpu_2node_8try/config.yaml', '2>', '/home/your_user_name/mpi_work/mnist_gpu_2node_8try/work/error/0.txt']",20230731-072335-485
1,1,g0006.abci.local,"start: recv: gpu_num=0",20230731-072335-487
1,1,g0006.abci.local,"_func_sub(): debug: line=Train Epoch: 1 [0/60000 (0%)]    Loss: 2.299825",20230731-072402-481
1,1,g0006.abci.local,"_func_sub(): debug: line=Train Epoch: 1 [640/60000 (1%)]  Loss: 1.391941",20230731-072402-482
1,1,g0006.abci.local,"_func_sub(): debug: line=Train Epoch: 1 [1280/60000 (2%)] Loss: 0.876705",20230731-072402-482
(中略)
1,1,g0006.abci.local,"_func_sub(): debug: line=Train Epoch: 14 [59520/60000 (99%)]      Loss: 0.003460",20230731-072638-273
1,1,g0006.abci.local,"_func_sub(): debug: line=Test set: Average loss: 0.0260, Accuracy: 9907/10000 (99%)",20230731-072639-726
1,1,g0006.abci.local,"_func_sub(): debug: line=0.02604724588394165",20230731-072640-650
1,1,g0006.abci.local,"_func_sub(): debug: line=before break: s=|| save_line=|0.02604724588394165|",20230731-072640-651
1,1,g0006.abci.local,"_func_sub(): debug: line=end: save_line=|0.02604724588394165| ret_s=|0.02604724588394165|",20230731-072640-651
1,1,g0006.abci.local,"_func_sub(): debug: line=0.02604724588394165",20230731-072640-652
1,1,g0006.abci.local,"_func_sub(): end: tag=1 process finished.",20230731-072640-823
~~~
