#! /bin/bash

set -e

wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
/bin/bash Miniforge3-Linux-x86_64.sh -b -p ./miniforge3

eval "$(./miniforge3/bin/conda shell.bash hook 2> /dev/null)"

mamba install -y \
    cython ruff mypy progressbar2 rich==14.0.0 tqdm \
    numpy scipy scikit-learn jupyter pandas matplotlib

pip3 install torch torchvision
