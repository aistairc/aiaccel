# Example of Black Box Optimization on ABCI 3.0

This is an example of performing black-box optimization of the learning rate for a ResNet50 model on the CIFAR10 dataset.

## Getting started

In an environment where aiaccel is installed, additionally install pyproject.toml in examples/torch/image_classification.

```bash
cd examples/torch/image_classification
pip install .
```

PATH_TO_ENV and JOB_GROUP in job_config.yaml should be changed to the path of the environment prepared above and the job group id.

```yaml
path_to_env: PATH_TO_ENV
job_group: JOB_GROUP
```

Run the following command to perform black-box optimization.

```bash
cmd=aiaccel-job pbs --config config_job.yaml

aiaccel-hpo optimize --config config_hpo.yaml -- \
    $cmd train --n_gpus=1 {config.working_directory}/{job_name}.log -- \
        aiaccel-torch train config_torch.yaml \
            working_directory={config.working_directory}/{job_name}/ \
            task.optimizer_config.optimizer_generator.lr={lr} \
            out_filename={out_filename}
```

## Detailed Descriptions

The target function for optimization using aiaccel.hpo.app.optimize is objective_integration.main.
Within objective_integration.main, aiaccel.torch.app.train is called, and the learning rate is returned.

Detailed descriptions of torch and optimize are available on the [aiaccel document(torch)](https://aistairc.github.io/aiaccel/user_guide/torch.html) [aiaccel document(optimize)](https://aistairc.github.io/aiaccel/user_guide/hpo.html)
