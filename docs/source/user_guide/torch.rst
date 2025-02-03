Writing a PyTorch Training Script (WIP)
=======================================

Writing a simple training script
--------------------------------

Running inference
-----------------

To run aiaccel on ABCI3.0, you need an environment with Python 3.10.
This guide explains how to set up the environment using Singularity.

.. note::

    For details on how to use Singularity, please refer to the following documentation:
    https://docs.abci.ai/v3/en/containers/

Create the following Singularity definition file:

.. code-block:: bash
    :caption: aiaccel_env.def

    BootStrap: docker

    From: python:3.10

    %post

        pip install --upgrade pip

        # aiaccel env
        pip install aiaccel[torch]@git+https://github.com/aistairc/aiaccel.git@develop/v2

        # torch/MNIST example env
        pip install torchvision


Use the Singularity definition file to build a Singularity image file:

.. code-block:: bash

    singularity build aiaccel.sif aiaccel_env.def

Use the Singularity image file to execute aiaccel:

.. code-block:: bash

    singularity exec --nv aiaccel.sif python -m aiaccel.torch.apps.train $wd/config.yaml --working_directory $wd


Writing a DDP training script
-----------------------------

Accelerating your training
--------------------------
