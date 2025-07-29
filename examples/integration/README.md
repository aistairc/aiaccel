# Example of Black Box Optimization on ABCI 3.0

This is an example of performing black-box optimization of the learning rate for a ResNet50 model on the MNIST dataset.

## Getting started

In an environment where aiaccel is installed, additionally install torchvision.

```bash
pip install torchvision
```


Run the following command to perform black-box optimization.
PATH_TO_ENV should be changed to the path of the environment prepared above.

```bash
python -m aiaccel.hpo.apps.optimize "python -m aiaccel.jobs.cli.abci3 gpu --command_prefix 'cd \$PBS_O_WORKDIR && module load cuda/12.6/12.6.1 && module load python/3.13/3.13.2 && source PATH_TO_ENV/bin/activate &&' jobs/{job_name}.log -- python -m aiaccel.torch.apps.train resnet50/config.yaml task.optimizer_config.optimizer_generator.lr={lr} trainer.logger.name=lr_{lr} out_filename={out_filename}" --config config.yaml
```

## Detailed Descriptions

The target function for optimization using aiaccel.hpo.app.optimize is objective_integration.main.
Within objective_integration.main, aiaccel.torch.app.train is called, and the learning rate is returned.

Detailed descriptions of torch and optimize are available on the [aiaccel document(torch)](https://aistairc.github.io/aiaccel/user_guide/torch.html) [aiaccel document(optimize)](https://aistairc.github.io/aiaccel/user_guide/hpo.html)
