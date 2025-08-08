# Example of Black Box Optimization on ABCI 3.0

This is an example of performing black-box optimization of the learning rate for a ResNet50 model on the MNIST dataset.

## Getting started

In an environment where aiaccel is installed, additionally install pyproject.toml.

```bash
pip install .
```

PATH_TO_ENV in job_config.yaml should be changed to the path of the environment prepared above.

```yaml
    source PATH_TO_ENV
```

Run the following command to perform black-box optimization.

```bash
aiaccel-hpo optimize --config hpo_config.yaml -- \
    aiaccel-job local --config job_config.yaml train --n_gpus=1 jobs/{job_name}.log -- \
        aiaccel-torch train resnet50/config.yaml \
            working_directory=jobs/{job_name}/ \
            task.optimizer_config.optimizer_generator.lr={lr} \
            out_filename={out_filename}
```

## Detailed Descriptions

The target function for optimization using aiaccel.hpo.app.optimize is objective_integration.main.
Within objective_integration.main, aiaccel.torch.app.train is called, and the learning rate is returned.

Detailed descriptions of torch and optimize are available on the [aiaccel document(torch)](https://aistairc.github.io/aiaccel/user_guide/torch.html) [aiaccel document(optimize)](https://aistairc.github.io/aiaccel/user_guide/hpo.html)
