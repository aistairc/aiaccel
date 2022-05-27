#!/bin/bash

#$-l rt_F=1
#$-j y
#$-cwd
#$ -l h_rt=2:00:00

source /etc/profile.d/modules.sh
module load gcc/11.2.0 python/3.8/3.8.13 cuda/10.1/10.1.243 cudnn/7.6/7.6.5
source ../../../../resnet50sample_env/bin/activate
