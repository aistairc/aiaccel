#! /bin/bash

#PBS -q rt_HF
#PBS -l select=1:mpiprocs=192
#PBS -l walltime=1:00:00
#PBS -P grpname
#PBS -j oe

cd ${PBS_O_WORKDIR}

source /etc/profile.d/modules.sh
module load hpcx

wd=path_to_working_directory
num_gpus=$(nvidia-smi -L | wc -l)

mpirun -np $num_gpus -bind-to none -map-by slot \
        -x MAIN_ADDR=$(hostname -i)  \
        -x MAIN_PORT=3000 \
        singularity exec --nv aiaccel.sif python -m aiaccel.torch.apps.train $wd/config_ddp.yaml --working_directory $wd
