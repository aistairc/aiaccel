Optimizing Your Hyperparameters
===============================

Hyperparameter optimization (HPO) is an indispensable step to make it work in real world.


Usage
--------

Creating an Objective File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a file that defines the objective function to be optimized:

.. code-block:: python

    # objective.py
    from argparse import ArgumentParser
    from pathlib import Path
    import pickle as pkl


    def main() -> None:
        parser = ArgumentParser()
        parser.add_argument("dst_filename", type=Path)
        parser.add_argument("--x1", type=float)
        parser.add_argument("--x2", type=float)
        args = parser.parse_args()

        x1, x2 = args.x1, args.x2

        # Objective function: (x1^2 - 4x1 + x2^2 - x2 - x1x2)
        y = (x1**2) - (4.0 * x1) + (x2**2) - x2 - (x1 * x2)

        with open(args.dst_filename, "wb") as f:
            pkl.dump(y, f)


    if __name__ == "__main__":
        main()

Creating an Execution Script
~~~~~~~~~~~~~~~~~~~~

Create a script for environment setup and objective function execution:

.. code-block:: bash

    # objective.sh
    #!/bin/bash

    #$-l rt_C.small=1
    #$-cwd

    source /etc/profile.d/modules.sh
    module load gcc/13.2.0
    module load python/3.10/3.10.14

    python objective.py $@

Execution Methods
~~~~~~~~~~~~~~~~

Running with Command Line Parameters
++++++++++++++++++++++++++++

You can execute optimization with the following command:

.. code-block:: bash

    python -m aiaccel.hpo.apps.optimize objective.sh params.x1="[0,1]" params.x2="[0,1]" n_trials=30 n_max_jobs=4 group=gaa50000

Main parameters:

:params.x1, params.x2: Search range for each parameter
:n_trials: Number of optimization trials
:n_max_jobs: Maximum number of concurrent jobs
:group: ABCI group identifier (required when running on ABCI)

Running with Configuration File Parameters
+++++++++++++++++++++++++++

For more complex configurations, you can use a YAML configuration file:

.. code-block:: yaml

    # config.yaml
    params:
      x1: [0, 1]
      x2: [0, 1]

    n_trials: 30
    n_max_jobs: 4
    group: gaa50000

To execute using the configuration file:

.. code-block:: bash

    python -m aiaccel.hpo.apps.optimize objective.sh --config config.yaml



Combining Configuration File and Command Line Arguments
+++++++++++++++

You can combine configuration file and command line arguments. When the same parameter is specified in both places, command line arguments take precedence.


(WIP)
