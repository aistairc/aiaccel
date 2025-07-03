# Example of Black Box Optimization on ABCI 3.0

This is an example of performing black-box optimization of the learning rate for a ResNet50 model on the MNIST dataset.

## Getting started

In an environment where aiaccel is installed, additionally install TensorBoard and torchvision.

```bash
pip install tensorboard torchvision
```

Run the following command to perform black-box optimization.

```bash
python -m aiaccel.hpo.apps.optimize --config config_abci.yaml
```

## Detailed Descriptions
Detailed descriptions are available on the [aiaccel document(torch)](https://aistairc.github.io/aiaccel/user_guide/torch.html) [aiaccel document(optimize)](https://aistairc.github.io/aiaccel/user_guide/hpo.html)
