Optimizing Your Hyperparameters
===============================

Hyperparameter optimization (HPO) is an indispensable step to make it work in real
world.

Getting Started
---------------

Create a file that defines the objective function to be optimized:

.. code-block:: python
    :caption: objective.py

    def main(x1, x2) -> float:
        y = (x1**2) - (4.0 * x1) + (x2**2) - x2 - (x1 * x2)
        return y

Run the following command:

.. code-block:: bash

    python -m aiaccel.hpo.apps.optimize params.x1="[0,2]" params.x2="[0,2]" objective._target_="objective.main" n_trials=30

The parameters are set as params.x1="[0,2]" and params.x2="[0,2]", the target function
is specified with objective._target_="objective.main", and the number of trials is set
to n_trials=30.

Basic Usage
-----------

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

The study configuration controls the overall behavior of the optimization process:

.. code-block:: yaml

    study:
        _target_: optuna.create_study  # default
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
- CmaEsSampler: Covariance Matrix Adaptation Evolution Strategy (good for continuous
  parameters)
- GridSampler: Exhaustive grid search (for small parameter spaces)
- NSGAIISampler: For multi-objective optimization
- NelderMeadSampler: Nelder-Mead optimization

Cluster Configuration
~~~~~~~~~~~~~~~~~~~~~

The cluster section configures the distributed computing environment using
Dask.distributed library for parallel execution of hyperparameter optimization tasks:

.. code-block:: yaml

    cluster:
        _target_: distributed.Client  # default
        n_workers: 4  # Number of workers to start (default : 1)
        threads_per_worker: 1  # Number of threads per each worker  (default : 1)

Parameters Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

The parameters section defines the hyperparameter search space using Optuna's suggestion
methods wrapped by aiaccel:

.. code-block:: yaml

    params:
        _convert_: partial
        _target_: aiaccel.hpo.apps.optimize.HparamsManager  # default

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
~~~~~~~~~~~~~~~

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

.. code-block:: yaml

    n_trials: 100
    n_max_jobs: 1  # default : 1

Usage Examples
~~~~~~~~~~~~~~

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

    python -m aiaccel.hpo.apps.optimize --config config.yaml --resume params.x1="[0,2]"

HPO Using NelderMeadSampler
---------------------------

Basic Usage
~~~~~~~~~~~

Basic optimization example using NelderMeadSampler:

Search Space
++++++++++++

NelderMeadSampler requires a search space as an argument.

.. code-block:: python
    :caption: examples/hpo/samplers/example.py

    search_space = {
        "x": (-10.0, 10.0),
        "y": (-10.0, 10.0),
    }

Objective Function
++++++++++++++++++

Set the Objective Function in the same way as in regular Optuna. The optimization target
is the benchmark function Sphere.

.. code-block:: python
    :caption: examples/hpo/samplers/example.py

    def sphere(trial: optuna.trial.Trial) -> float:
        params = []
        for name, distribution in search_space.items():
            params.append(trial.suggest_float(name, *distribution))

        return float(np.sum(np.asarray(params) ** 2))

Execute Optimization
++++++++++++++++++++

Specify NelderMeadSampler as the sampler and execute the optimization.

.. code-block:: python
    :caption: examples/hpo/samplers/example.py

    study = optuna.create_study(
        sampler=NelderMeadSampler(search_space=search_space, seed=42)
    )
    study.optimize(func=sphere, n_trials=100)

Full code is examples/hpo/samplers/example.py

Pallarel Optimization
~~~~~~~~~~~~~~~~~~~~~

Example pallarel optimization:

.. code-block:: python
    :caption: examples/hpo/samplers/example_parallel.py

    study = optuna.create_study(
        sampler=NelderMeadSampler(search_space=search_space, seed=42, block=True)
    )
    study.optimize(func=sphere, n_trials=100, n_jobs=3)

Parallel execution is enabled by setting the NelderMeadSampler argument block=True and
the study.optimize argument n_jobs>2. By enabling parallel execution, the initial point
calculation and the computation during shrinking can be parallelized, leading to faster
execution compared to serial execution.

Full code is examples/hpo/samplers/example_parallel.py

optuna.study.enqueue_trial
~~~~~~~~~~~~~~~~~~~~~~~~~~

Example using optuna.study.enqueue_trial:

.. code-block:: python
    :caption: examples/hpo/samplers/example_enqueue.py

    study = optuna.create_study(
        sampler=NelderMeadSampler(search_space=search_space, seed=42)
    )
    study.enqueue_trial({"x": 1.0, "y": 1.0})
    study.enqueue_trial({"x": 1.0, "y": 2.0})
    study.enqueue_trial({"x": 2.0, "y": 1.0})
    study.optimize(func=sphere, n_trials=100)

Utilizing the ask-tell interface, random parameters are explored using enqueue_trial
when NelderMeadSampler fails to output parameters.

Full code is examples/hpo/samplers/example_enqueue.py

Sub Sampler
~~~~~~~~~~~

Example using sub_sampler as optuna.samplers.TPESampler:

.. code-block:: python
    :caption: examples/hpo/samplers/example_sub_sampler.py

    study = optuna.create_study(
        sampler=NelderMeadSampler(
            search_space=search_space,
            seed=42,
            sub_sampler=optuna.samplers.TPESampler(seed=42),
        )
    )
    study.optimize(func=sphere, n_trials=100, n_jobs=3)

When sub_sampler=optuna.samplers.TPESampler is set as an argument for NelderMeadSampler,
TPESampler is used for exploration when NelderMeadSampler fails to output parameters.
When using the sub_sampler function, the argument block=False must be set even if it is
parallel. (Parallel execution is possible even with block=False.)

Full code is examples/hpo/samplers/example_sub_sampler.py
