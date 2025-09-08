Managing Configurations
=======================

This guide introduces how to manage configuration files using ``aiaccel.config`` and
`Hydra's instantiation mechanism
<https://hydra.cc/docs/advanced/instantiate_objects/overview/>`_. The key features of
``aiaccel.config`` are:

- Modular programming through YAML meta-programming
- Efficient management of multiple config files using ``_base_`` and ``_inherit_``
  attributes
- Easy version control integration with Git
- Minimal dependency on Hydra (only uses ``hydra.utils.instantiate``)

Getting Started
---------------

Aiaccel's configuration system is based on `OmegaConf
<http://omegaconf.readthedocs.io/>`_. The typical usage is:

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


    overwrite_omegaconf_dumper()

    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Config file in YAML format")
    args, unk_args = parser.parse_known_args()

    config = load_config(args.config)
    print_config(config)
    config = resolve_inherit(config)

    model = instantiate(config.model)

    print(model)

    ...

To run the script:

.. code-block:: bash

    python example.py config.yaml

``load_config`` reads the configuration file and processes the ``_base_`` attribute,
while ``resolve_inherit`` resolves ``_inherit_`` attributes.

``_base_`` and ``_inherit_``
----------------------------

The ``_base_`` attribute allows you to inherit from another configuration file.

Example base configuration:

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

Example configuration that uses a base:

.. code-block:: yaml
    :caption: config.yaml

    _base_: config_base.yaml
    n_trials: 100
    n_max_jobs: 4

``config.yaml`` is automatically expanded to include the contents of
```config_base.yaml``.

The ``_inherit_`` attribute, on the other hand, allows you to duplicate and modify parts
of the configuration. Example configuration:

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

After processing, the configuration will be expanded so that ``x1`` and ``x2`` each
include the contents of ``param`` along with their own ``name`` fields.

Version Controlling
-------------------

WIP

Additional Information
----------------------

Detailed information is available at :doc:`API Reference <../api_reference/config>`.
