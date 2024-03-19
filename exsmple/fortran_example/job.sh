#!/bin/bash

#$-l rt_C.small=1
#$-cwd

source /etc/profile.d/modules.sh
module load gcc/12.2.0

gfortran objective.f95
./a.out $@
