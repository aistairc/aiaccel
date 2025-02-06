#! /bin/bash

#PBS -q rt_HF
#PBS -l select=1
#PBS -l walltime=1:00:00
#PBS -P grpname
#PBS -j oe

cd ${PBS_O_WORKDIR}

source /etc/profile.d/modules.sh
module load cuda/12.6/12.6.1

wd=path_to_working_directory

singularity exec --nv path_to_python.sif python -m aiaccel.torch.apps.train $wd/config_ddp.yaml --working_directory $wd
