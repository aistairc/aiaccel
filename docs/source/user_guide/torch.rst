Training a PyTorch Model
========================

``aiaccel.torch`` is built to make PyTorch training modular based on `PyTorch Lightning
<https://lightning.ai/docs/pytorch/stable/>`_ and quick to iterate: you describe
datasets, schedulers, and trainers in YAML, keep access to the underlying APIs, and can
ship the exact same config from laptops to clusters. This page
walks through the design, shows how to assemble configs, and highlights the features
that accelerate multi-GPU runs or job submissions.

Core Concepts
-------------

``aiaccel-torch`` is designed around the following key concepts:

- Keep the toolkit modular; datasets, pipelines, and Lightning helpers remain optional
  pieces, so you can import just :mod:`aiaccel.torch.h5py` or
  :mod:`aiaccel.torch.datasets` without pulling in Lightning at all.
- Provide ``aiaccel-torch train`` to hide the repetitive parts of training scripts
  (config loading, accelerator selection, checkpointing) while still exposing the full
  Lightning + Hydra stack for customization.
- Treat HPC scenarios as first-class by bundling helpers such as dataset caching
  in :mod:`aiaccel.torch.datasets` and the HDF5 utilities in :mod:`aiaccel.torch.h5py`
  so you can author fast training loops that fully utilize compute resources on shared clusters.

Basic Usage
-----------

Start by invoking the CLI so the workflow feels concrete:

.. code-block:: bash

    aiaccel-torch train config.yaml trainer.max_epochs=30

The command loads ``config.yaml`` through :func:`aiaccel.config.load_config`, merges any
``key=value`` overrides, resolves ``_inherit_`` entries, and instantiates the trainer,
task, and datamodule via `hydra.utils.instantiate
<https://hydra.cc/docs/advanced/instantiate_objects/overview/>`_ before calling
`lightning.Trainer.fit()
<https://lightning.ai/docs/pytorch/stable/common/trainer.html#basic-use>`_.
Whenever ``trainer.is_global_zero`` is ``True`` the
merged YAML is saved to ``working_directory/merged_config.yaml``. Because overrides are
parsed by `OmegaConf.from_cli
<https://omegaconf.readthedocs.io/en/latest/usage.html#command-line-flags>`_, changing
values such as
``datamodule.batch_size=256`` mirrors the workflow shown in :doc:`config`.

Composing the config
~~~~~~~~~~~~~~~~~~~~

A minimal configuration extends ``train_base.yaml`` (under
:mod:`aiaccel.torch.apps.config`) and defines ``trainer``, ``datamodule``, and ``task``.
Each block maps directly to Hydra instantiation arguments, so any Lightning callback or
dataset factory can be wired in-place.

.. code-block:: yaml
    :caption: config.yaml

    _base_: ${resolve_pkg_path:aiaccel.torch.apps.config}/train_base.yaml

    trainer:
      max_epochs: 10
      callbacks:
        - _target_: lightning.pytorch.callbacks.ModelCheckpoint
          filename: "{epoch:04d}"
          save_last: true
          save_top_k: -1

    datamodule:
      _target_: aiaccel.torch.lightning.datamodules.SingleDataModule
      batch_size: 128
      train_dataset_fn:
        _partial_: true
        _target_: torchvision.datasets.MNIST
        root: ./dataset
        train: true
        download: true
      val_dataset_fn:
        _partial_: true
        _inherit_: ${datamodule.train_dataset_fn}
        train: false

    task:
      _target_: my_project.tasks.MyTask
      num_classes: 10
      optimizer_config:
        _target_: aiaccel.torch.lightning.OptimizerConfig
        optimizer_generator:
          _partial_: true
          _target_: torch.optim.Adam
          lr: 1e-4

Switching to DDP and launching jobs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``train_ddp.yaml`` extends ``train_base.yaml`` with synchronized BatchNorm, the
:class:`aiaccel.torch.lightning.abci_environment.ABCIEnvironment` plugin, and automatic
``devices`` / ``num_nodes`` detection from MPI variables. Enable it by stacking bases so
the remainder of the file stays intact:

.. code-block:: yaml
    :caption: Switching to DDP

    _base_:
      - ${resolve_pkg_path:aiaccel.torch.apps.config}/train_base.yaml
      - ${resolve_pkg_path:aiaccel.torch.apps.config}/train_ddp.yaml

Any values that follow the base list override the combined template, so callbacks,
datasets, and optimizers continue to live in the same file. Once the config is ready,
wrap the command with ``aiaccel-job`` to request GPUs from a scheduler:

.. code-block:: bash

    aiaccel-job local train --n_gpus=8 logs/train.log -- \
        aiaccel-torch train configs/config.yaml

The job YAML (see :doc:`job`) handles queue-specific options, while the ``aiaccel.torch``
config only toggles between single-node and DDP behavior through ``_base_``. Moving to
``pbs`` or ``sge`` swaps the backend name but keeps the command payload exactly the
same.

Advanced Topics
---------------

Understanding the modules
~~~~~~~~~~~~~~~~~~~~~~~~~

The torch package exposes multiple namespaces that can be mixed and matched:

- ``aiaccel.torch.lightning`` ships :class:`OptimizerConfig`, reusable LightningModule
  scaffolds, pipeline helpers, and :class:`~aiaccel.torch.lightning.datamodules.SingleDataModule`
  to keep trainer setups declarative.
- ``aiaccel.torch.lr_schedulers`` provides schedulers such as
  :class:`aiaccel.torch.lr_schedulers.SequentialLR` so complex learning-rate plans can
  be authored in YAML.
- ``aiaccel.torch.datasets`` adds caching wrappers, scatter helpers, and utilities for
  slicing datasets when running through array jobs or MPI workers.
- ``aiaccel.torch.h5py`` contains :class:`aiaccel.torch.h5py.HDF5Writer` for preparing
  and exporting large intermediate results that other dataloaders can stream.

Further reading
---------------

- :doc:`../api_reference/torch` for the full API reference of datasets, pipelines,
  Lightning helpers, schedulers, and I/O utilities.
- :doc:`config` for composition patterns, resolvers, and CLI overrides used by torch
  configs.
- :doc:`job` for the job-launching interface that wraps the ``aiaccel-torch`` command.
