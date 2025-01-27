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
 import json
 from argparse import ArgumentParser
 from pathlib import Path

 def main() -> None:
   parser = ArgumentParser()
   parser.add_argument("dst_filename", type=Path)
   parser.add_argument("--x1", type=float)
   parser.add_argument("--x2", type=float)
   args = parser.parse_args()

   x1, x2 = args.x1, args.x2

   # Objective function: (x1^2 - 4x1 + x2^2 - x2 - x1x2)
   y = (x1**2) - (4.0 * x1) + (x2**2) - x2 - (x1 * x2)

   with open(args.dst_filename, 'w') as f:
     json.dump(y, f)

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

Configuration
~~~~~~~~~~~~~~~~~~

Basic configuration example:

.. code-block:: yaml

 study:
   _target_: optuna.create_study
   direction: minimize

 executor:
   _target_: aiaccel.hpo.job_executors.LocalJobExecutor
   n_max_jobs: 4

 result:
   _target_: aiaccel.results.JsonJobOutputLoader
   filename_template: "{job.cwd}/{job.job_name}_result.json"

 params:
   x1: [0, 1]
   x2: [0, 1]

 n_trials: 30

Study Configuration
~~~~~~~~~~~~~~~~~~~~

(WIP)

Sampler Configuration
~~~~~~~~~~~~~~~~~~~~

(WIP)

Executor Configuration
~~~~~~~~~~~~~~~~~~~~~

Two types of executors are available:

Local Execution:

.. code-block:: yaml

 executor:
   _target_: aiaccel.hpo.job_executors.LocalJobExecutor
   n_max_jobs: 4

ABCI Execution:

.. code-block:: yaml

 executor:
   _target_: aiaccel.hpo.job_executors.AbciJobExecutor
   n_max_jobs: 4
   group: gaa50000

Result Configuration
~~~~~~~~~~~~~~~~~~~

Three formats are supported for saving results:

JSON Format (Default):

.. code-block:: yaml

 result:
   _target_: aiaccel.hpo.job_output_loaders.JsonJobOutputLoader
   filename_template: "{job.cwd}/{job.job_name}_result.json"

Example objective function for JSON:

.. code-block:: python

 import json
 ...
 def main() -> None:
   # ... argument parsing ...
   y = (x1**2) - (4.0 * x1) + (x2**2) - x2 - (x1 * x2)
   with open(args.dst_filename, 'w') as f:
     json.dump(y, f)

Pickle Format:

.. code-block:: yaml

 result:
   _target_: aiaccel.hpo.job_output_loaders.PickleJobOutputLoader
   filename_template: "{job.cwd}/{job.job_name}_result.pkl"

Example objective function for Pickle:

.. code-block:: python

 import pickle
 ...
 def main() -> None:
   # ... argument parsing ...
   y = (x1**2) - (4.0 * x1) + (x2**2) - x2 - (x1 * x2)
   with open(args.dst_filename, 'wb') as f:
     pickle.dump(y, f)

Stdout Format:

.. code-block:: yaml

 result:
   _target_: aiaccel.results.StdoutJobOutputLoader
   filename_template: "{job.cwd}/{job.job_name}_result.txt"

Example objective function for Stdout:

.. code-block:: python

 ...
 def main() -> None:
   # ... argument parsing ...
   y = (x1**2) - (4.0 * x1) + (x2**2) - x2 - (x1 * x2)
   print(y)

Parameters Configuration
~~~~~~~~~~~~~~~~~~~~~~~

(WIP)

Execution Methods
~~~~~~~~~~~~~~~~

Basic Usage
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
