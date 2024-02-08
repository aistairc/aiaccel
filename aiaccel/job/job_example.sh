#!/bin/bash

#$ -l rt_F=4
#$ -l h_rt=01:00:00
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.11
module load hpcx-mt/2.12

mpiexec -n 4 -hostfile ./hostfile python user_program.py -e --params x=3.745401188473625
