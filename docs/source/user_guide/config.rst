Managing Configurations
=======================

``aiaccel.config`` is a lightweight utility built on top of `Hydra's instantiation
mechanism <https://hydra.cc/docs/advanced/instantiate_objects/overview/>`_ and
`OmegaConf <http://omegaconf.readthedocs.io/>`_. It keeps machine learning experiment
configurations compact, composable, and easy to reuse. This page walks through the core
design ideas and explains each feature step by step.

Core Concepts
-------------

The configuration system centers on the following principles:

- Compose configuration fragments through YAML meta-programming
- Organize multiple files hierarchically via ``_base_`` / ``_inherit_``
- Depend only on `hydra.utils.instantiate
  <https://hydra.cc/docs/advanced/instantiate_objects/overview/>`_ to keep the runtime
  coupling minimal
- Integrate with ``git`` to inspect the state of referenced Python packages

All of these utilities live under the ``aiaccel.config`` namespace.

Basic Usage
-----------

The typical workflow is to combine :func:`aiaccel.config.load_config` and
:func:`aiaccel.config.resolve_inherit` to construct the final config, optionally merge
command-line overrides, then instantiate objects from ``_target_`` definitions via
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
        load_config,
        overwrite_omegaconf_dumper,
        print_config,
        resolve_inherit,
    )
    from hydra.utils import instantiate
    from omegaconf import OmegaConf as oc  # noqa: N813


    overwrite_omegaconf_dumper()

    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Config file in YAML format")
    args, unk_args = parser.parse_known_args()

    config = load_config(args.config)
    print_config(config)
    config = resolve_inherit(config)
    config = oc.merge(config, oc.from_cli(unk_args))

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

:func:`aiaccel.config.load_config` reads the file and resolves ``_base_`` entries, while
:func:`aiaccel.config.resolve_inherit` expands all ``_inherit_`` references. Because the
parser captures ``unk_args`` separately, you can override any value from the command
line by appending ``key=value`` pairs, and ``oc.merge`` combines them after every other
transformation.

Advanced Topics
---------------

Using ``_base_`` to inherit files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Here ``config_base.yaml`` is loaded first and ``config.yaml`` overwrites values such as
``n_trials``. Because :func:`aiaccel.config.load_config` resolves ``_base_``
recursively, base files are free to declare further bases of their own.

Reusing fragments with ``_inherit_``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``_inherit_`` copies arbitrary DictConfig nodes inline. It is useful for sharing
repeated parameter definitions in a single place.

.. code-block:: yaml
    :caption: config.yaml

    params:
      _convert_: partial
      _target_: aiaccel.hpo.optuna.hparams_manager.HparamsManager
      x1:
        _inherit_: "${param}"
      x2:
        _inherit_: "${param}"

    objective:
      _target_: objective.main

    n_trials: 30
    n_max_jobs: 4

    param:
      _target_: aiaccel.hpo.optuna.hparams.Float
      low: 0.0
      high: 1.0
      log: false

:func:`aiaccel.config.resolve_inherit` makes ``params.x1`` and ``params.x2`` contain the
fields declared under ``param`` and can further override them locally. Passing a list of
references merges multiple templates in sequence.

Resolvers registered by :func:`aiaccel.config.load_config`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every call to :func:`aiaccel.config.load_config` registers the following resolvers in
OmegaConf:

- ``eval``: safe arithmetic evaluation powered by `simpleeval
  <https://github.com/danthedeckie/simpleeval>`_. Example: ``n_trials:
  ${eval:"${n_max_jobs} * 10"}``
- ``resolve_pkg_path``: proxies `importlib.resources.files
  <https://docs.python.org/3/library/importlib.html#importlib.resources.files>`_ to
  locate the on-disk path of a Python package. Example: ``dataset_root:
  ${resolve_pkg_path:aiaccel}/examples/data``

They are always registered with ``oc.register_new_resolver(..., replace=True)``, so
aiaccel's resolvers take precedence even if other code defined the same names.

Git status checks
~~~~~~~~~~~~~~~~~

Whenever a config references a Python package via ``_target_``,
:func:`aiaccel.config.collect_git_status_from_config` detects it and collects ``git
status`` / ``git rev-parse`` results for each repository. The associated
:meth:`aiaccel.config.PackageGitStatus.ready` helper reports whether there are
uncommitted changes. Packages that are not Git repositories or that are ignored via
``.gitignore`` are skipped automatically.

CLI Utilities
-------------

``aiaccel/config/apps/`` ships small CLIs to streamline common config operations.

``check_git``
    Inspect the Git status of every package referenced by ``_target_``.

    .. code-block:: bash

        aiaccel-config check_git config.yaml

    When changes are detected, the tool prints the output of
    :func:`aiaccel.config.print_git_status` and exits with status ``1``.

``get_value``
    Fetch a key using `OmegaConf's select
    <https://omegaconf.readthedocs.io/en/latest/usage.html#access-and-traversal>`_
    syntax, which is handy when you need the value after ``_inherit_`` has been fully
    resolved.

    .. code-block:: bash

        aiaccel-config get_value config.yaml params.x1.low

Further reading
---------------

See :doc:`../api_reference/config` for the full API reference.
