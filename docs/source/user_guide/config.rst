OmegaConf Utilities
===================

Typical usage is as follows:

.. code-block:: python

    from argparse import ArgumentParser

    from omegaconf import OmegaConf as oc  # noqa: N813
    from aiaccel.config import (
        load_config,
        overwrite_omegaconf_dumper,
        pathlib2str_config,
        print_config,
        resolve_inherit,
    )

    overwrite_omegaconf_dumper()

    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Config file in YAML format")
    args, unk_args = parser.parse_known_args()

    config = oc.merge(
        load_config(
            args.config,
            {
                "config_path": args.config,
                "working_directory": str(Path(args.config).parent.resolve()),
            },
        ),
        oc.from_cli(unk_args),
    )

    print_config(config)

    config = resolve_inherit(config)

Basic Config File
-----------------

A typical config file for aiaccel is structured as follows:

.. code-block:: yaml
    :caption: config.yaml

    params:
        _convert_: partial
        _target_: aiaccel.hpo.apps.optimize.HparamsManager
        x1: [0, 1]
        x2:
            _target_: aiaccel.hpo.optuna.suggest_wrapper.SuggestFloat
            name: x2
            low: 0.0
            high: 1.0
            log: false
    n_trials: 30
    n_max_jobs: 4

This setup assumes the use of hydra.utils.instantiate. Functions are specified and
loaded using the _target_ key. For more details, please refer to the documentation for
hydra.utils.instantiate.

Usage of _base_ element
-----------------------

You can specify a base configuration using _base_.

.. code-block:: yaml
    :caption: config_base.yaml

    params:
        _convert_: partial
        _target_: aiaccel.hpo.apps.optimize.HparamsManager
        x1: [0, 1]
        x2:
            _target_: aiaccel.hpo.optuna.suggest_wrapper.SuggestFloat
            name: x2
            low: 0.0
            high: 1.0
            log: false

.. code-block:: yaml
    :caption: config.yaml

    _base_: config_base.yaml
    n_trials: 100
    n_max_jobs: 4

When loading the config.yaml above, it will be expanded as follows:

.. code-block:: yaml

    params:
        _convert_: partial
        _target_: aiaccel.hpo.apps.optimize.HparamsManager
        x1: [0, 1]
        x2:
            _target_: aiaccel.hpo.optuna.suggest_wrapper.SuggestFloat
            name: x2
            low: 0.0
            high: 1.0
            log: false
    n_trials: 100
    n_max_jobs: 4

Usage of _inherit_ element
--------------------------

You can use _inherit_ to copy other elements:

.. code-block:: yaml
    :caption: config.yaml

    params:
        _convert_: partial
        _target_: aiaccel.hpo.apps.optimize.HparamsManager
        x1: {$param}
        x2: {$param}
    n_trials: 30
    n_max_jobs: 4

    param: [0, 1]

When loading the config.yaml above, it will be expanded as follows:

.. code-block:: yaml

    params:
        _convert_: partial
        _target_: aiaccel.hpo.apps.optimize.HparamsManager
        x1: [0, 1]
        x2: [0, 1]
    n_trials: 30
    n_max_jobs: 4

    param: [0, 1]
