Optimizing Your Hyperparameters
===============================

Hyperparameter optimization (HPO) is an indispensable step to make it work in real
world.

Usage
-----

Creating an Objective File
~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a file that defines the objective function to be optimized:

.. code-block:: python
    # objective.py

    def main(x1, x2) -> float:
        y = (x1**2) - (4.0 * x1) + (x2**2) - x2 - (x1 * x2)
        return y


Configuration
~~~~~~~~~~~~~

Basic configuration example:

.. code-block:: yaml
    study:
        _target_: optuna.create_study
        direction: minimize
        storage: # Set this item if results need to be stored in DB
            _target_: optuna.storages.RDBStorage
            url: sqlite:///aiaccel_storage.db
            engine_kwargs:
                connect_args:
                timeout: 30
        study_name: my_study  # Set this item if results need to be stored in DB
        sampler:
            _target_: optuna.samplers.TPESampler
            seed: 0

    cluster:
        _target_: distributed.Client
        n_workers: 4
        threads_per_worker: 1

    params:
        x1: [0, 1]
        x2: [0, 1]

    n_trials: 30
    n_max_jobs: 1


Study Configuration
~~~~~~~~~~~~~~~~~~~

(WIP)

The study configuration controls the overall behavior of the optimization process:

.. code-block:: yaml
    study:
        _target_: optuna.create_study
        direction: minimize     # 'minimize' or 'maximize' depending on your objective
        study_name: my_study    # Name of the study (optional)
        storage:  # This item is not required. This item is not required if there is no need to record it in the file.
            _target_: optuna.storages.RDBStorage
            url: sqlite:///example.db
            engine_kwargs:
                connect_args:
                    timeout: 30
    load_if_exists: true    # Load existing study if it exists
    sampler:
        _target_: optuna.samplers.TPESampler
        seed: 42

Sampler Configuration
~~~~~~~~~~~~~~~~~~~~~

(WIP)
The sampler determines the algorithm used to search the hyperparameter space:

.. code-block:: yaml
    study:
        _target_: optuna.create_study
    direction: minimize
    sampler:
        _target_: optuna.samplers.TPESampler  # Tree-structured Parzen Estimator (default)
        # TPE-specific parameter
        seed: 42                           # For reproducibility
        n_startup_trials: 10               # Number of random trials before using TPE

Available samplers include:

- TPESampler: Efficient Bayesian optimization approach (recommended for most cases)
- RandomSampler: Simple random search (useful as baseline)
- CmaEsSampler: Covariance Matrix Adaptation Evolution Strategy (good for continuous parameters)
- GridSampler: Exhaustive grid search (for small parameter spaces)
- NSGAIISampler: For multi-objective optimization
- NelderMeadSampler: Nelder-Mead optimization


Cluster Configuration
~~~~~~~~~~~~~~~~~~~~~

(WIP)

The cluster section configures the distributed computing environment using Dask.distributed library for parallel execution of hyperparameter optimization tasks:

.. code-block:: yaml
    cluster:
        _target_: distributed.Client
        n_workers: 4
        threads_per_worker: 1


Parameters Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

(WIP)

The parameters section defines the hyperparameter search space using Optuna's suggestion methods wrapped by aiaccel:

.. code-block:: yaml
    params:
        _convert_: partial
        _target_: aiaccel.hpo.apps.optimize.HparamsManager

    # Float parameter example
    x1:
        _target_: aiaccel.hpo.optuna.suggest_wrapper.SuggestFloat
        name: x1
        low: 0.0
        high: 1.0
        log: false

    # Another float parameter
    x2:
        _target_: aiaccel.hpo.optuna.suggest_wrapper.SuggestFloat
        name: x2
        low: 0.0
        high: 1.0
        log: false

    # Shorthand for float parameters
    x3: [0, 1]

Parameter Types
~~~~~~~~~~~~~~~~

aiaccel supports multiple parameter types through different suggestion wrappers:

- SuggestFloat: For continuous parameters
.. code-block:: yaml
    learning_rate:
        _target_: aiaccel.hpo.optuna.suggest_wrapper.SuggestFloat
        name: learning_rate
        low: 0.0001
        high: 0.1
        log: true  # Use logarithmic scale for learning rates

- SuggestInt: For integer parameters
.. code-block:: yaml
    num_layers:
        _target_: aiaccel.hpo.optuna.suggest_wrapper.SuggestInt
        name: num_layers
        low: 1
        high: 10

- SuggestCategorical: For categorical parameters
.. code-block:: yaml
    optimizer:
        _target_: aiaccel.hpo.optuna.suggest_wrapper.SuggestCategorical
        name: optimizer
        choices: ['adam', 'sgd', 'rmsprop']

- SuggestDiscreteUniform: For discrete uniform parameters
.. code-block:: yaml
    batch_size:
        _target_: aiaccel.hpo.optuna.suggest_wrapper.SuggestDiscreteUniform
        name: batch_size
        low: 32
        high: 256
        q: 32

- SuggestLogUniform: For log-uniform parameters
.. code-block:: yaml
    learning_rate:
        _target_: aiaccel.hpo.optuna.suggest_wrapper.SuggestLogUniform
        name: learning_rate
        low: 0.0001
        high: 0.1

- SuggestLogInt: For log-int parameters
.. code-block:: yaml
    num_layers:
        _target_: aiaccel.hpo.optuna.suggest_wrapper.SuggestLogInt
        name: num_layers
        low: 1
        high: 10


Objective Function
~~~~~~~~~~~~~~~~~~

The objective function is the main function to be optimized:

.. code-block:: yaml
    objective:
        _target_: objective.main

Other Configuration Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- n_trials: Number of trials to run
- n_max_jobs: Maximum number of parallel jobs


Usage Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here are some common usage patterns:

Start a new study:

.. code-block:: bash
    python -m aiaccel.hpo.apps.optimize --config config.yaml

Resume from the previous study:

.. code-block:: bash
    python -m aiaccel.hpo.apps.optimize --config config.yaml --resume

Make the study resumable (sets appropriate storage configuration):

.. code-block:: bash
    python -m aiaccel.hpo.apps.optimize --config config.yaml --resumable

Resume a study and override parameters:

.. code-block:: bash
    python -m aiaccel.hpo.apps.optimize --config config.yaml --resume --params x1="[0,2]"
