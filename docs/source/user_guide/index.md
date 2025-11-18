# User Guide

## Installation
You can install aiaccel directly from PyPI:
```bash
python -m pip install aiaccel
```

Alternatively, we provide an example environment that includes all dependencies:
```bash
git clone https://github.com/aistairc/aiaccel/  # clone aiaccel repository
cd aiaccel/

pushd examples/python/
bash setup.sh  # setup a miniforge-based Python environment
source activate.sh  # activate the Python environment
popd
```

## Tutorials
```{toctree}
:maxdepth: 1
config
job
torch
hpo
```