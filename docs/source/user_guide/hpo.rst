############################
 Optimizing Hyperparameters
############################

``aiaccel-hpo`` wraps Optuna so you can launch reproducible hyperparameter optimization
jobs with the same configuration system used by ``aiaccel-job`` and ``aiaccel-torch``.
This page summarizes the workflow implemented in :mod:`aiaccel.hpo.apps.optimize` and
shows how to describe the search space using :mod:`aiaccel.config`.

Core Concepts
-------------

- ``aiaccel-hpo optimize`` builds an Optuna :class:`~optuna.study.Study`, manages
  multiple workers with :class:`concurrent.futures.ThreadPoolExecutor`, and streams
  objective results via JSON files (:mod:`aiaccel.hpo.apps.optimize`).
- Configuration files follow the exact same Hydra-inspired syntax explained in
  :doc:`config`; overrides passed before ``--`` are merged through
  :func:`omegaconf.OmegaConf.from_cli`.
- Parameters are declared via :class:`aiaccel.hpo.optuna.hparams_manager.HparamsManager`
  so the search space is instantiated and validated before the first trial begins.
- Every run writes the fully merged config to ``{working_directory}/merged_config.yaml``
  and stores Optuna state in ``optuna.db`` by default
  (:mod:`aiaccel.hpo.apps.config.default`).

Basic Usage
-----------

Create an objective script that consumes hyperparameters and writes a JSON-compatible
scalar or list to ``out_filename``:

.. code-block:: python
    :caption: objective.py

    import argparse
    import json


    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("out_filename", type=str)
        parser.add_argument("--x1", type=float)
        parser.add_argument("--x2", type=float)
        args = parser.parse_args()

        y = (args.x1**2) - (4.0 * args.x1) + (args.x2**2) - args.x2 - (args.x1 * args.x2)

        with open(args.out_filename, "w") as f:
            json.dump(y, f)


    if __name__ == "__main__":
        main()

Launch optimization directly from the command line by supplying parameter overrides
before ``--`` and the objective command after it:

.. code-block:: bash

    python -m aiaccel.hpo.apps.optimize \
        params.x1="[0,2]" params.x2="[0,2]" n_trials=30 n_max_jobs=2 \
        -- python ./objective.py --x1={x1} --x2={x2} {out_filename}

``aiaccel-hpo`` expands ``{x1}``, ``{x2}``, ``{out_filename}``, and ``{job_name}``
placeholders when launching each worker. The script reads the JSON file, feeds the value
back to :meth:`optuna.study.Study.tell`, deletes the file, and continues submitting
trials until ``n_trials`` is reached.

Under the hood
~~~~~~~~~~~~~~

The CLI is intentionally thin: :mod:`aiaccel.hpo.apps.optimize` loads the YAML through
:func:`aiaccel.config.load_config`, merges ``key=value`` overrides with
:func:`omegaconf.OmegaConf.from_cli`, resolves ``_inherit_`` entries, and merely
instantiates two objects—``config.study`` and ``config.params``—via
:func:`hydra.utils.instantiate`. The study is a vanilla Optuna object (any sampler or
storage backend works), while ``config.params`` becomes an
:class:`aiaccel.hpo.optuna.hparams_manager.HparamsManager` that generates the
``{param}`` placeholders. After that, the module just loops over ``study.ask`` /
``study.tell`` while running your command in a thread pool. No additional abstractions
are introduced, so whatever Optuna supports can be expressed directly in YAML.

Configuring optimizer behavior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can configure the behavior of ``aiaccel-hpo`` in detail by authoring a
``config.yaml``. Using `aiaccel/hpo/apps/config/default.yaml
<https://github.com/aistairc/aiaccel/blob/main/aiaccel/hpo/apps/config/default.yaml>`_
as a base keeps the layout familiar while you fine-tune studies without stuffing the CLI
with overrides:

.. code-block:: yaml

    db_filename: ${working_directory}/optuna.db

    n_trials: 100
    n_max_jobs: 10

    study:
      _target_: optuna.create_study
      study_name: aiaccel-hpo
      storage:
        _target_: optuna.storages.RDBStorage
        url: sqlite:///${db_filename}
      load_if_exists: True

    params:
      _convert_: partial
      _target_: aiaccel.hpo.optuna.hparams_manager.HparamsManager

Extend this file or inherit from it via ``_base_`` to describe your objective.

.. list-table::
   :widths: 20 35 45
   :header-rows: 1

   - - Scope
     - Key fields
     - Notes
   - - ``study``
     - ``direction``, ``sampler``, ``storage``, ``study_name``, ``load_if_exists``
     - Passed directly to :func:`optuna.create_study`, so you can swap samplers or
       storage backends without touching the Python code.
   - - ``params``
     - Literal ``[low, high]`` pairs, constants, or ``_target_`` entries
     - Instantiates :class:`aiaccel.hpo.optuna.hparams_manager.HparamsManager`; each
       child becomes a :class:`aiaccel.hpo.optuna.hparams.Hparam`` feeding values to
       the command template.
   - - ``command``
     - Token list such as ``["python", "train.py", "--lr={lr}", "{out_filename}"]``
     - ``aiaccel-hpo`` joins the list, interpolates ``{param}``, ``{out_filename}``,
       ``{working_directory}``, ``{job_name}``, and runs it via :func:`subprocess.run`.
   - - Run control
     - ``n_trials``, ``n_max_jobs``, ``working_directory``, ``db_filename``
     - Limit total evaluations, cap concurrent workers, choose the artifact root, and
       point Optuna to the database used for resuming studies.

Search spaces rely on the helpers in :mod:`aiaccel.hpo.optuna.hparams`
(:class:`Float`, :class:`Int`, :class:`Categorical`, :class:`Const`). Because they are
dataclasses, you can define reusable fragments in the same YAML and pull them in with
``_inherit_`` when multiple parameters share the same range or log-scale behavior.

Writing a ``config.yaml`` lets you capture these choices once, reuse them across runs,
and keep the CLI invocation short—only pass overrides for the few values that change
per experiment (e.g. ``n_trials=200`` or ``params.lr.low=1e-5``). The CLI prints the
merged config path and working directory so you can revisit or resume later.

Practical reminders for day-to-day runs:

- Treat ``{out_filename}`` as write-only; it is removed immediately after the JSON is
  read.
- Store datasets or checkpoints beneath ``working_directory`` when possible so each run
  stays self-contained.
- Point Optuna's visualization or dashboard tools at ``{working_directory}/optuna.db``
  to inspect intermediate results and resume safely.

Advanced Topics
---------------

Using Nelder-Mead samplers
~~~~~~~~~~~~~~~~~~~~~~~~~~

The Nelder-Mead sampler exposed in :class:`~aiaccel.hpo.optuna.samplers.NelderMeadSampler`
targets scenarios where gradients are unavailable, evaluation cost is high, and the
search space is moderate in dimensionality. By evolving a simplex rather than relying
on probabilistic surrogates it:

- excels on smooth, low-dimensional objectives where each evaluation is expensive and
  noisy gradients would derail deterministic optimizers;
- explores parameter combinations deterministically, making resume/replay runs easier
  to reason about than adaptive stochastic samplers;
- integrates naturally with enqueue-based warm starts, letting you seed the simplex with
  domain knowledge or previously discovered points;
- provides a queue-aware implementation so the expensive initial simplex evaluation and
  shrink steps can run in parallel when ``n_max_jobs`` > 1.

These traits make Nelder-Mead a strong fit for simulation-heavy or hardware-in-the-loop
use cases where the number of tunable knobs is small but each trial is costly. Refer to
the dedicated documentation for usage details and configuration options.

Further Reading
---------------

- :doc:`user_guide/config` - complete reference for Hydra-style YAML composition used
  by ``aiaccel-hpo``.
- :doc:`user_guide/job` - explains how payload commands are embedded in templates, which
  mirrors the ``command`` handling in the optimizer.
- ``examples/hpo`` - runnable Optuna examples, including Nelder-Mead flows and COCO
  benchmarks.
- Optuna documentation - visualization utilities, sampler details, and storage backends
  that extend what ``aiaccel-hpo`` instantiates for you.
