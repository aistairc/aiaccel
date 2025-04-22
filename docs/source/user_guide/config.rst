OmegaConf Utilities
===================

Typical Usage
-------------

Typical usage is as follows:

.. code-block:: python
    :caption: example_config.py

    from argparse import ArgumentParser

    from aiaccel.config import (
        load_config,
        overwrite_omegaconf_dumper,
        print_config,
        resolve_inherit,
    )

    overwrite_omegaconf_dumper()

    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Config file in YAML format")
    args, unk_args = parser.parse_known_args()

    config = load_config(args.config)

    print_config(config)

    config = resolve_inherit(config)

    print_config(config)

We use OmegaConf and hydra.utils.instantiate. OmegaConf is mainly used for loading
configuration files. hydra.utils.instantiate is used to handle functions defined within
the configuration.

When you run the sample code above, load_config handles configuration loading and
processes the _base_ attribute, while resolve_inherit processes the _inherit_ attribute.

.. code-block:: bash

    python example_config.py --config config.yaml

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

Usage of _base_ attribute
-------------------------

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

Usage of _inherit_ attribute
----------------------------

You can use _inherit_ to copy other elements:

.. code-block:: yaml
    :caption: config.yaml

    params:
        _convert_: partial
        _target_: aiaccel.hpo.apps.optimize.HparamsManager
        x1:
            _inherit_: "${param}"
            name: x1
        x2:
            _inherit_: "${param}"
            name: x2

    objective:
        _target_: objective.main

    n_trials: 30
    n_max_jobs: 4

    param:
        _target_: aiaccel.hpo.optuna.suggest_wrapper.SuggestFloat
        low: 0.0
        high: 1.0
        log: false

When loading the config.yaml above, it will be expanded as follows:

.. code-block:: yaml

    params:
        _convert_: partial
        _target_: aiaccel.hpo.apps.optimize.HparamsManager
        x1:
            _target_: aiaccel.hpo.optuna.suggest_wrapper.SuggestFloat
            low: 0.0
            high: 1.0
            log: false
            name: x1
        x2:
            _target_: aiaccel.hpo.optuna.suggest_wrapper.SuggestFloat
            low: 0.0
            high: 1.0
            log: false
            name: x2

    objective:
        _target_: objective.main

    n_trials: 30
    n_max_jobs: 4

    param:
        _target_: aiaccel.hpo.optuna.suggest_wrapper.SuggestFloat
        low: 0.0
        high: 1.0
        log: false
