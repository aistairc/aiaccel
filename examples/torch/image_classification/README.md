# Training a ResNet50 on CIFAR-10

## Setup
We assume the Python-environment setup at `examples/python`.
```bash
pushd ../../python/
bash setup_python.sh
. activate.sh
popd
```

```bash
pip install -e .
```

## Training on a single GPU
```bash
qsub -I -P [group_name] -q rt_HG -l select=1 -l walltime=1:0:0

cd $PBS_O_WORKDIR
. ../../python/activate.sh

cd recipes
aiaccel-torch train resnet50.cifar10/config.yaml
```

## Training on on multiple GPUs
This script will automatically use all the GPUs in your computer. The hyperparameter is assumed to use eight GPUs.
```bash
qsub -I -P [group_name] -q rt_HF -l select=1 -l walltime=1:0:0

cd $PBS_O_WORKDIR
. ../../python/activate.sh

cd recipes
aiaccel-torch train resnet50.cifar10.ddp/config.yaml
```

## Training with the Makefile
The example also provides a `Makefile` that prepares the CIFAR-10 dataset, generates a run config under `runs/`, and launches training via `aiaccel-job`.

```bash
make help
```

This Makefile assumes the ABCI environment and an already activated Python virtual environment.
Please run it either with your virtual environment already activated, or by specifying env_activate_command.
ABCI-specific job settings such as `walltime`, `qsub`, and `env_activate_command` are defined in `recipes/job_config.yaml`.
The default mode is `single`, so use `make all` for single-GPU training and `make all mode=ddp` for multi-GPU training.

```bash
source /path/to/env_aiaccel/bin/activate
make all
```

```bash
make all env_activate_command="source /path/to/env_aiaccel/bin/activate"
```

### Single GPU
```bash
make all
```

### Multiple GPUs (DDP)
```bash
make all mode=ddp
```

## Detailed Descriptions
Detailed descriptions are available on the [aiaccel document](https://aistairc.github.io/aiaccel/user_guide/torch.html)
