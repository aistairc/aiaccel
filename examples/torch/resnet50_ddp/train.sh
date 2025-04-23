#! /bin/bash

#PBS -q rt_HF
#PBS -l select=1:mpiprocs=8:ompthreads=12
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -k oed

cd ${PBS_O_WORKDIR}

source /etc/profile.d/modules.sh

module load cuda/12.6/12.6.1
module load python/3.13/3.13.2
module load hpcx

mpirun -bind-to none -map-by slot \
    -mca pml ob1 -mca btl self,tcp -mca btl_tcp_if_include bond0 \\
    -x MAIN_ADDR=$(hostname -i)  \
    -x MAIN_PORT=3000 \
    python -m aiaccel.torch.apps.train resnet50_ddp/config.yaml
