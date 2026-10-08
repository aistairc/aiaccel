#########################
 Managing Configurations
#########################

``aiaccel.config`` is a lightweight utility built on top of `Hydra's instantiation
mechanism <https://hydra.cc/docs/advanced/instantiate_objects/overview/>`_ and
`OmegaConf <http://omegaconf.readthedocs.io/>`_. It keeps machine-learning experiment
configurations compact, composable, and easy to reuse. The configuration system centers
on the following principles:

- Keep experiment setup in YAML so models, datasets, and runtime parameters can be
  reviewed and reproduced without editing Python code
- Reuse configuration across files and within a file via ``_base_`` and ``_inherit_``,
  while still allowing ``key=value`` command-line overrides for per-run changes
- Turn the resolved config into the Python objects needed at runtime, using
  `hydra.utils.instantiate
  <https://hydra.cc/docs/advanced/instantiate_objects/overview/>`_ to generate models,
  datasets, optimizers, and other components from YAML

All of these utilities live under the ``aiaccel.config`` namespace.

*************
 Basic Usage
*************

The typical workflow is to call :func:`~aiaccel.config.prepare_config`, optionally merge
command-line overrides, and then instantiate objects from ``_target_`` definitions via
`hydra.utils.instantiate
<https://hydra.cc/docs/advanced/instantiate_objects/overview/>`_.

.. code-block:: yaml
    :caption: config.yaml

    model:
      _target_: torchvision.models.resnet50
      num_classes: 13

.. code-block:: python
    :caption: example.py

    from argparse import ArgumentParser

    from aiaccel.config import (
        prepare_config,
        print_config,
    )
    from hydra.utils import instantiate
    from omegaconf import OmegaConf as oc  # noqa: N813

    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Config file in YAML format")
    args, unk_args = parser.parse_known_args()

    config = prepare_config(
        args.config,
        overwrite_config=oc.from_cli(unk_args),
    )
    print_config(config)  # optional; useful while iterating on configs

    model = instantiate(config.model)
    print(model)

Run the script with:

.. code-block:: bash

    python example.py config.yaml

``OmegaConf.from_cli`` `documented here
<https://omegaconf.readthedocs.io/en/latest/usage.html#command-line-flags>`_ allows
overrides such as

.. code-block:: bash

    python example.py config.yaml model.num_classes=20

:func:`~aiaccel.config.prepare_config` is the high-level entry point. Internally it:

- calls :func:`~aiaccel.config.setup_omegaconf` to register aiaccel's custom resolvers;
- calls :func:`~aiaccel.config.load_config` to read the YAML file and resolve ``_base_``
  recursively;
- merges ``overwrite_config`` if provided;
- calls :func:`~aiaccel.config.resolve_inherit` to expand ``_inherit_`` references;
- optionally prints or saves the fully merged config.

Use :func:`~aiaccel.config.load_config` directly only when you need low-level control
over the loading step before inheritance resolution.

Using ``_base_`` to inherit files
=================================

``_base_`` lets you inherit from one or more YAML files while overriding their values.
When multiple files are provided, they are merged in the order given.

.. code-block:: yaml
    :caption: config_base.yaml

    params:
      _convert_: partial
      _target_: aiaccel.hpo.optuna.hparams_manager.HparamsManager
      x1: [0, 1]
      x2:
        _target_: aiaccel.hpo.optuna.hparams.Float
        low: 0.0
        high: 1.0
        log: false

.. code-block:: yaml
    :caption: config.yaml

    _base_: config_base.yaml
    n_trials: 100
    n_max_jobs: 4

Here ``config_base.yaml`` is loaded first and ``config.yaml`` overwrites any overlapping
values. Because :func:`~aiaccel.config.load_config` resolves ``_base_`` recursively,
base files can themselves declare additional bases.

The ``_replace_`` attribute replaces the entire mapping at that node after ``_base_``
has been resolved. Use it when you want to discard the inherited keys under a section
and define that section from scratch.

Example configuration:

.. code-block:: yaml
    :caption: config_base.yaml

    params:
      _convert_: partial
      _target_: aiaccel.hpo.optuna.hparams_manager.HparamsManager
      x1: [0, 10]
      x2: [10, 20]

.. code-block:: yaml
    :caption: config.yaml

    _base_: config_base.yaml

    params:
      _replace_: true
      _convert_: partial
      _target_: aiaccel.hpo.optuna.hparams_manager.HparamsManager
      x1: [0, 1]

    n_trials: 100
    n_max_jobs: 4

After loading, ``params`` contains only ``x1``. The ``x2`` entry from
``config_base.yaml`` is dropped because ``params`` is replaced as a whole instead of
being merged with the base mapping.

Reusing fragments with ``_inherit_``
====================================

``_inherit_`` copies arbitrary DictConfig nodes inline. It is useful for sharing
repeated parameter definitions in a single place while overriding only the fields that
need to change.

.. code-block:: yaml
    :caption: config.yaml

    params:
      _convert_: partial
      _target_: aiaccel.hpo.optuna.hparams_manager.HparamsManager
      x1:
        _inherit_: "${param}"
        high: 1.0
      x2:
        _inherit_: "${param}"
        high: 10.0
        log: true

    objective:
      _target_: objective.main

    n_trials: 30
    n_max_jobs: 4

    param:
      _target_: aiaccel.hpo.optuna.hparams.Float
      low: 0.0
      high: 5.0
      log: false

:func:`~aiaccel.config.resolve_inherit` makes ``params.x1`` and ``params.x2`` contain
the fields declared under ``param``, then applies any local overrides. In this example,
``params.x1`` becomes a ``Float(low=0.0, high=1.0, log=False)``-style config, while
``params.x2`` reuses the same template but overrides ``high`` to ``10.0`` and ``log`` to
``true``. Passing a list of references merges multiple templates in sequence.

Resolvers registered by :func:`~aiaccel.config.setup_omegaconf`
===============================================================

:func:`~aiaccel.config.setup_omegaconf` registers the following resolvers in OmegaConf:

- ``eval``: safe arithmetic evaluation powered by `simpleeval
  <https://github.com/danthedeckie/simpleeval>`_. Example: ``n_trials:
  ${eval:"${n_max_jobs} * 10"}``
- ``resolve_pkg_path``: proxies `importlib.resources.files
  <https://docs.python.org/3/library/importlib.html#importlib.resources.files>`_ to
  locate the on-disk path of a Python package. Example: ``dataset_root:
  ${resolve_pkg_path:aiaccel}/examples/data``

The setup is idempotent, so calling :func:`~aiaccel.config.prepare_config` or
:func:`~aiaccel.config.print_config` repeatedly is safe. Internally the resolvers are
registered with ``oc.register_new_resolver(..., replace=True)``, so aiaccel's
definitions take precedence even if the same names were registered elsewhere.

*****************
 Advanced Topics
*****************

Git status checks
=================

Whenever a config references a Python package via ``_target_``,
:func:`~aiaccel.config.collect_git_status_from_config` detects it and collects ``git
status`` / ``git rev-parse`` results for each repository. The associated
:meth:`aiaccel.config.PackageGitStatus.ready` helper reports whether there are
uncommitted changes. Packages that are not Git repositories, or that are ignored via
``.gitignore`` are skipped automatically.

***************
 CLI Utilities
***************

``aiaccel/config/apps/`` ships small CLIs to streamline common config operations.

``check_git``
    Inspect the Git status of every package referenced by ``_target_``.

    .. code-block:: bash

        aiaccel-config check_git config.yaml

    When changes are detected, the tool prints the output of
    :func:`~aiaccel.config.print_git_status` and exits with status ``1``.

``get_value``
    Fetch a key using `OmegaConf's select
    <https://omegaconf.readthedocs.io/en/latest/usage.html#access-and-traversal>`_
    syntax, which is handy when you need a value after ``_inherit_`` has been fully
    resolved.

    .. code-block:: bash

        aiaccel-config get_value config.yaml params.x1.low

*****************
 Further reading
*****************

See :doc:`../api_reference/config` for the full API reference.
