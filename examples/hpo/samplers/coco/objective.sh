#!/bin/bash

#$-l rt_C.small=1
#$-cwd

source /etc/profile.d/modules.sh
module load gcc/13.2.0
module load python/3.10/3.10.14
source xxx/aiaccel_env/bin/activate

python3.10 experiment_coco.py $@
