# インストール方法

## ~/mpienvを作成。(aiaccelとmpi4pyをインストール。)
~~~
ssh abci
qrsh -g group-name -l rt_C.small=1 -l h_rt=2:00:00
source /etc/profile.d/modules.sh
module load gcc/11.2.0
module load python/3.8/3.8.13
module load openmpi/4.1.3
python3 -m venv mpienv
source mpienv/bin/activate
pip install git+https://github.com/aistairc/aiaccel.git
pip install mpi4py
deactivate
exit
~~~

上記作業時のaiaccelのバージョンは
~~
commit d297a5203b973b299ced5c4b2a95756702721670
Date:   Fri Dec 16 13:28:46 2022 +0900
~~

## ソースコードを~/aiaccelに展開。

ソースコード(src/aiaccel.tgz)がabciの~にあることが前提。
~~~
ssh abci
tar zxf ~/aiaccel.tgz
~~~

## gpuテストのための~/tfenvを作成。(kerasのmnist_convnetをインストール。)
~~~
ssh abci
qrsh -g group-name -l rt_G.small=1 -l h_rt=2:00:00
source /etc/profile.d/modules.sh
module load gcc/11.2.0
module load python/3.8/3.8.13
module load cuda/11.0/11.0.3
module load cudnn/8.0/8.0.5
python3 -m venv ~/tfenv
source ~/tfenv/bin/activate
pip3 install --upgrade pip setuptools
pip3 install tensorflow==2.4.1
git clone https://github.com/keras-team/keras-io.git
deactivate
exit
~~~

### gpuテストのためのkeras-io/mnist_convnet.pyの動作確認。
~~~
ssh abci
qrsh -g group-name -l rt_G.small=1 -l h_rt=2:00:00
source /etc/profile.d/modules.sh
module load gcc/11.2.0
module load python/3.8/3.8.13
module load cuda/11.0/11.0.3
module load cudnn/8.0/8.0.5
source ~/tfenv/bin/activate
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
python3 ~/keras-io/examples/vision/mnist_convnet.py
~~~

# 使用方法

## cpuでの使用方法

### qrsh(cpu)での使用方法
~~~
ssh abci
qrsh -g group-name -l rt_C.small=1 -l h_rt=2:00:00
cat $SGE_JOB_HOSTLIST
source /etc/profile.d/modules.sh
module load gcc/11.2.0
module load python/3.8/3.8.13
module load openmpi/4.1.3
source ~/mpienv/bin/activate
export PYTHONPATH=~/aiaccel
cd ~/aiaccel/examples/experimental/mpi/sphere
python3 -m aiaccel.experimental.mpi.cli.start --config config.yaml --make_hostfile
cat hostfile
#mpiexec -n 4 -npernode 4 \
mpiexec -n 5 -hostfile hostfile \
python3 -m mpi4py.futures -m aiaccel.experimental.mpi.cli.start --config config.yaml --clean --from_mpi_bat
~~~

### qsub(cpu)での使用方法
~~~
ssh abci
source /etc/profile.d/modules.sh
module load gcc/11.2.0
module load python/3.8/3.8.13
source ~/mpienv/bin/activate
export PYTHONPATH=~/aiaccel
cd ~/aiaccel/examples/experimental/mpi/sphere
python3 -m aiaccel.experimental.mpi.cli.start --config config.yaml --clean
~~~

### gpuでの使用方法

### qrsh(gpu)での使用方法
~~~
ssh abci
qrsh -g group-name -l rt_F=1 -l h_rt=2:00:00
cat $SGE_JOB_HOSTLIST
source /etc/profile.d/modules.sh
module load gcc/11.2.0
module load python/3.8/3.8.13
module load openmpi/4.1.3
source ~/mpienv/bin/activate
export PYTHONPATH=~/aiaccel
cd ~/aiaccel/examples/experimental/mpi/mnist_convnet
python3 -m aiaccel.experimental.mpi.cli.start --config config.yaml --make_hostfile
cat hostfile
mpiexec -n 5 -hostfile hostfile \
python3 -m mpi4py.futures -m aiaccel.experimental.mpi.cli.start --config config.yaml --clean --from_mpi_bat
~~~

### qsub(gpu)での使用方法
~~~
ssh abci
source /etc/profile.d/modules.sh
module load gcc/11.2.0
module load python/3.8/3.8.13
source ~/mpienv/bin/activate
export PYTHONPATH=~/aiaccel
cd ~/aiaccel/examples/experimental/mpi/mnist_convnet
python3 -m aiaccel.experimental.mpi.cli.start --config config.yaml --clean
~~~

# その他の機能、情報など

## qsub先でのnvidia-smiなどのコマンドの実行方法
~~~
python3 -m aiaccel.experimental.mpi.cli.qrsh-inherit 0 'nvidia-smi'
python3 -m aiaccel.experimental.mpi.cli.qrsh-inherit 0 'ps -ef | grep acb'
~~~

## 集中管理のprotocolなど
~~~
start ---

main() cli.start (aiaccel.experimeental.mpi.)
Mpi.run_main() util.mpi (aiaccel.experimeental.mpi.)
workspace.create() workspace (aiaccel.)
Mpi.prepare(workspace.path)
loop_1
# Mpi.abort()

loop_1 ---

rank 0

before_job_submitted(obj) MpiModel scheduler.job.model.mpi_model (aiaccel.experimeental.mpi.)
(processor, tag) = submit(runner_command) util.mpi (aiaccel.experimeental.mpi.)
obj.th_oh = MpiOutputHandler(processor, tag) util.mpi (aiaccel.experimeental.mpi.)

submit() Mpi util.mpi (aiaccel.experimeental.mpi.)
cls.tag += 1 (first cls.tag=0)
cls.executor.submit(cls._func, command, cls.tag, str(cls.rank_log_path))
02 [rank, processor] = recv(tag=cls.tag)
03 send(gpu_num, rank) if gpu_mode

run() MpiOutputHandler util.mpi (aiaccel.experimeental.mpi.)
loop_2
06 line = recv(tag=self._tag)
if line.find(Mpi.func_end_id) == 0:
    break

rank n(!=0)

_func(command, tag, rank_log_path_str) Mpi util.mpi (aiaccel.experimeental.mpi.)
01 send([rank, processor], 0, tag=tag)
04 gpu_num = recv(source=0) if gpu_mode
loop_3
05 send(line, 0, tag=tag)
send(f'{cls.func_end_id} process finished.') if end
(send(f'{cls.func_end_id} error.') if error)
~~~
