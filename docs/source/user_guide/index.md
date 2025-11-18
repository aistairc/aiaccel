# User Guide

## Installation
You can install aiaccel directly from PyPI:
```bash
python -m pip install aiaccel
```

## Tutorials
When you want to try the tutorials, we recommend setting up the environment with `pixi`, which installs aiaccel together with every required dependency. You can first install `pixi` by following the instructions at https://pixi.sh/latest/#installation.

```bash
git clone https://github.com/aistairc/aiaccel.git
cd aiaccel
pixi install
pixi shell  # only if you want to enter the environment interactively
```

```{toctree}
:maxdepth: 1
config
torch
hpo
```
