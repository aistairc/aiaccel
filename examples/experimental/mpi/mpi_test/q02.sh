#!/bin/bash

#$ -l rt_F=2
#$ -l h_rt=1:00:00
#$ -j y
#$ -cwd

source /etc/profile.d/modules.sh
module load python/3.11/3.11.2
module load hpcx-mt/2.12
source ~/mpi_work/mpienv/bin/activate
mpiexec -n 5 -npernode 3 -hostfile $SGE_JOB_HOSTLIST python -m mpi4py.futures ~/mpi_work/aiaccel/examples/experimental/mpi/mpi_test/02.py
