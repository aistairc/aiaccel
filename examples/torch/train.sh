#! /bin/bash

#PBS -q rt_HG
#PBS -l select=1
#PBS -l walltime=1:00:00
#PBS -P grpname
#PBS -j oe
#PBS -k oed

cd ${PBS_O_WORKDIR}

source /etc/profile.d/modules.sh
module load cuda/12.6/12.6.1
module load python/3.13/3.13.2
source path_to_venv/bin/activate

wd=path_to_working_directory

python -m aiaccel.torch.apps.train $wd/config.yaml --working_directory $wd
