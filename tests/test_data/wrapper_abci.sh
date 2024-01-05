#!/bin/bash

#$-l rt_C.small=1
#$-cwd

source /etc/profile.d/modules.sh
source ~/work/bin/activate
module load python/3.6/3.6.5 singularity/2.6.1

AISTOPTPATH=$HOME/local/aiaccel-dev
export PYTHONPATH=$AISTOPTPATH:$AISTOPTPATH/lib
