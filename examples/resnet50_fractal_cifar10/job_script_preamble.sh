#!/bin/bash

#$-l rt_F=1
#$-j y
#$-cwd
#$ -l h_rt=8:00:00

source /etc/profile.d/modules.sh
module load gcc/12.2.0 python/3.10/3.10.10 cuda/12.2/12.2.0 cudnn/8.9/8.9.2
source ./work/bin/activate
cd FractalDB-Pretrained-ResNet-PyTorch
