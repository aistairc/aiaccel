#!/bin/bash

module load python/3.11/3.11.2
#module load hpcx/2.12
module load hpcx-mt/2.12
source ~/mpi_work/mpienv/bin/activate
cd ~/mpi_work/aiaccel/examples/experimental/mpi/mpi_test
python 01.py
