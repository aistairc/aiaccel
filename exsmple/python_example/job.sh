#!/bin/bash

#$-l rt_C.small=1
#$-cwd
#$-g gaa50001

source /etc/profile.d/modules.sh
module load gcc/12.2.0
module load python/3.10/3.10.10
module load cuda/11.8
module load cudnn/8.6

python objective.py $@
