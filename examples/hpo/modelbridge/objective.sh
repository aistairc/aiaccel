#!/bin/bash
#$-l rt_C.small=1
#$-cwd

# source /etc/profile.d/modules.sh
# module load python/3.10

python3 examples/hpo/modelbridge/objectives/simple_objective.py "$@"
