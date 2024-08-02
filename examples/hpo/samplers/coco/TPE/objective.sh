#!/bin/bash

#$-l rt_C.small=1
#$-cwd
#$ -l h_rt=8:00:00

source /etc/profile.d/modules.sh
module load gcc/13.2.0
module load python/3.10/3.10.14 
source /home/aac12958eq/hpopt/aiaccelv2/aiaccel_env/bin/activate

# python3.10 experiment_for_nm_sampler_parallel.py $@
python3.10 experiment_for_TPE_sampler_parallel.py $@
