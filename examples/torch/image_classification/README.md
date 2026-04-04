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

## Detailed Descriptions [TDB]
Detailed descriptions are available on the [aiaccel document](https://aistairc.github.io/aiaccel/user_guide/torch.html)