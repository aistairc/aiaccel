#!/bin/bash

#$-l rt_C.small=1
#$-j y
#$-cwd

source /etc/profile.d/modules.sh
module load gcc/11.2.0
module load python/3.8/3.8.13
source /path/to/optenv/bin/activate

AIACCELPATH=$HOME/local/aiaccel-dev
export PYTHONPATH=$AIACCELPATH:$AIACCELPATH/lib
