#!/bin/bash

#$-l rt_C.small=1
#$-cwd

source /etc/profile.d/modules.sh
module load gcc/12.2.0
module load python/3.10/3.10.10

python objective.py $@
