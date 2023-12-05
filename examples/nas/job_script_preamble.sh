#!/bin/bash

#$-l rt_F=3
#$-j y
#$-cwd
#$ -l h_rt=72:00:00

source /etc/profile.d/modules.sh
module load gcc/12.2.0 python/3.11/3.11.2 cuda/12.3/12.3.0 cudnn/8.9/8.9.5
source ~/optenv/bin/activate
