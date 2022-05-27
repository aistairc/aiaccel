#!/bin/bash

#$-l rt_C.small=1
#$-j y
#$-cwd

source /etc/profile.d/modules.sh
module load gcc/11.2.0
module load python/3.8/3.8.13 
module load cuda/10.2
module load cudnn/8.0/8.0.5
module load nccl/2.8/2.8.4-1 
source ~/optenv/bin/activate

AIACCELPATH=$HOME/local/aiaccel-dev
export PYTHONPATH=$AIACCELPATH:$AIACCELPATH/lib
