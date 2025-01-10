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
~~~~~~~~~~~~~~~~~~

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

Basic Configuration
~~~~~~~~~~~~~~~~~~

The basic configuration is the same for both local and ABCI environments, except for the executor settings:

.. code-block:: yaml

    # Configuration for the optimization study
    study:
      _target_: optuna.create_study
      direction: minimize

    # Parameter configuration
    params:
      _convert_: partial
      _target_: aiaccel.hpo.apps.optimize.HparamsManager
      x1: [0, 1]
      x2: [0, 1]

    n_trials: 30

    # Executor configuration - choose one based on your environment
    # For local execution:
    executor:
      _target_: aiaccel.hpo.job_executors.LocalJobExecutor
      n_max_jobs: 4

    # For ABCI execution:
    executor:
      _target_: aiaccel.hpo.job_executors.AbciJobExecutor
      n_max_jobs: 4
      group: gaa50000  # Required for ABCI

Key components of the configuration:

1. **study**: Defines the optimization study
   - `direction`: Whether to minimize or maximize the objective function

2. **params**: Defines hyperparameters to optimize
   - Simple range specification: `parameter: [min, max]`
   - Detailed configuration using `SuggestFloat`, `SuggestInt`, etc.
   - Supports various parameter types and distributions

3. **executor**: Configures job execution environment
   - Choose `LocalJobExecutor` for local machine or `AbciJobExecutor` for ABCI
   - `n_max_jobs`: Maximum number of concurrent jobs
   - `group`: Required for ABCI environment

4. **n_trials**: Total number of optimization trials to perform

For more advanced configurations, you can also specify:

.. code-block:: yaml

    # Storage configuration for study persistence
    study:
      _target_: optuna.create_study
      direction: minimize
      storage: sqlite:///study.db   # Database for storing results
      study_name: my_study         # Name of the study
      load_if_exists: true         # Resume existing study if present

    # More complex parameter configuration
    params:
      _convert_: partial
      _target_: aiaccel.hpo.apps.optimize.HparamsManager
      x1: 
        _target_: aiaccel.hpo.apps.optimize.SuggestFloat
        name: x1
        low: 0.0
        high: 1.0
        log: true     # Log scale sampling
      x2:
        _target_: aiaccel.hpo.apps.optimize.SuggestInt
        name: x2
        low: 1
        high: 100
        step: 2       # Step size for integer parameter

Execution Methods
~~~~~~~~~~~~~~~~

Running with Command Line Parameters
++++++++++++++++++++++++++++

You can execute optimization with the following command:

.. code-block:: bash

    python -m aiaccel.hpo.apps.optimize objective.sh --params x1="[0,1]" x2="[0,1]" n_trials=30 n_max_jobs=4

For ABCI environment, add the group parameter:

.. code-block:: bash

    python -m aiaccel.hpo.apps.optimize objective.sh --params x1="[0,1]" x2="[0,1]" n_trials=30 n_max_jobs=4 group=gaa50000

Running with Configuration File
+++++++++++++++++++++++++++

For more complex configurations, you can use a YAML configuration file:

.. code-block:: bash

    python -m aiaccel.hpo.apps.optimize objective.sh --config config.yaml

Combining Configuration File and Command Line Parameters
+++++++++++++++++++++++++++

You can override configuration file settings using command line parameters. Command line parameters take precedence over configuration file values:

.. code-block:: bash

    # Override parameters from config file
    python -m aiaccel.hpo.apps.optimize objective.sh --config config.yaml --params x1="[0,2]" x2="[0,2]"

    # Override other settings
    python -m aiaccel.hpo.apps.optimize objective.sh --config config.yaml n_trials=50

(WIP)
