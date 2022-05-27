#!/bin/bash

#$-l rt_C.small=1
#$-j y
#$-cwd

source /etc/profile.d/modules.sh
source ~/work/bin/activate
module load python/3.6/3.6.5 singularity/2.6.1

AIACCELPATH=$HOME/local/aiaccel-dev
export PYTHONPATH=$AIACCELPATH:$AIACCELPATH/lib
