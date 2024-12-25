#! /bin/bash

#$-l rt_F=1
#$-l h_rt=1:00:00
#$-j y
#$-cwd

source /etc/profile.d/modules.sh
module load singularitypro
module load hpcx/2.12

python -m aiaccel.torch.apps.train $wd/config.yaml --working_directory $wd
