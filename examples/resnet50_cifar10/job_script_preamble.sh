#!/bin/bash

#$-l rt_F=1
#$-cwd
#$ -l h_rt=2:00:00

source /etc/profile.d/modules.sh
module load gcc/8.5.0 python/3.10 cuda/11.8 cudnn/8.7
source ~/optenv/bin/activate
