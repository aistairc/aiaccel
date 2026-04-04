############################
 Optimizing Hyperparameters
############################

``aiaccel-hpo`` wraps Optuna so you can launch reproducible hyperparameter optimization
jobs with the same configuration system used by ``aiaccel-job`` and ``aiaccel-torch``.
Three design ideas matter most when reading the rest of this guide:

- Keep the optimizer thin. ``aiaccel-hpo optimize`` mainly prepares the config,
  instantiates ``config.study`` and ``config.params``, and then runs an ordinary Optuna
  ``study.ask`` / ``study.tell`` loop.
- Describe both the search space and the execution command in YAML, using the same
  override mechanism described in :doc:`config`.
- Exchange trial results through files so the objective can stay a plain command-line
  program.

This page summarizes the workflow implemented in :mod:`aiaccel.hpo.apps.optimize` and
shows how to describe the search space using :mod:`aiaccel.config`.

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

Internally, ``aiaccel-hpo optimize`` is roughly equivalent to the following:

.. code-block:: python
    :caption: Simplified optimization flow

    if args.config is None:
        config_filename = default_config_path
        working_directory = generated_working_directory
    else:
        config_filename = args.config
        working_directory = args.config.parent.resolve()

    config = prepare_config(
        config_filename=config_filename,
        working_directory=working_directory,
        overwrite_config=oc.from_cli(oc_args),
    )

    if len(args.command) > 0:
        config.command = args.command

    print_config(config)
    with open(config.working_directory / "merged_config.yaml", "w") as f:
        oc.save(pathlib2str_config(config), f)

    study = instantiate(config.study)
    params = instantiate(config.params)

    with ThreadPoolExecutor(config.n_max_jobs) as pool:
        while finished_job_count < config.n_trials:
            trial = study.ask()
            out_filename = config.working_directory / f"trial_{trial.number:0>6}.json"

            pool.submit(
                subprocess.run,
                shlex.join(config.command).format(
                    out_filename=out_filename,
                    job_name=f"trial_{trial.number:0>6}",
                    **params.suggest_hparams(trial),
                ),
                shell=True,
                check=True,
            )

            y = json.load(open(out_filename))
            out_filename.unlink()
            study.tell(trial, y)

``aiaccel-hpo`` expands placeholders such as ``{x1}``, ``{x2}``, ``{out_filename}``,
``{job_name}``, and ``{working_directory}`` when launching each worker. The objective
script writes its result to ``out_filename``, and the optimizer feeds that JSON value
back to :meth:`optuna.study.Study.tell`.

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
per experiment (e.g. ``n_trials=200`` or ``params.lr.low=1e-5``). The merged config is
saved under ``working_directory``, so each run remains easy to inspect or resume later.

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
