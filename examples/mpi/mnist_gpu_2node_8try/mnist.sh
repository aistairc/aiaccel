#!/bin/bash

source /etc/profile.d/modules.sh
module load python/3.11
module load cuda/11.8
module load cudnn/8.6
source ~/mpi_work/ptenv/bin/activate
python main.py --lr $1 # --download --epochs 1
deactivate
