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

The command loads ``config.yaml`` through :func:`~aiaccel.config.load_config`, merges any
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
:class:`~aiaccel.torch.lightning.abci_environment.ABCIEnvironment` plugin, and automatic
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

    aiaccel-job local train --n_gpus=8 my_model/train.log -- \
        aiaccel-torch train my_model/config.yaml

The job YAML (see :doc:`job`) handles queue-specific options, while the ``aiaccel.torch``
config only toggles between single-node and DDP behavior through ``_base_``. Moving to
``pbs`` or ``sge`` swaps the backend name but keeps the command payload exactly the
same.

Advanced Topics
---------------

Using ``CachedDataset`` for HPC storage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~aiaccel.torch.datasets.CachedDataset` keeps samples in a multiprocessing shared
memory manager so workers on the same node can reuse decoded tensors without hitting
the underlying filesystem. Wrap the original dataset directly and the wrapper will
handle caching automatically:

.. code-block:: yaml
    :caption: CachedDataset example

    datamodule:
      _target_: aiaccel.torch.lightning.datamodules.SingleDataModule
      batch_size: 256
      train_dataset_fn:
        _partial_: true
        _target_: aiaccel.torch.datasets.CachedDataset
        dataset:
          _target_: torchvision.datasets.ImageNet
          root: /mnt/datasets
          split: train

The first time ``__getitem__`` runs for a given index, ``CachedDataset`` stores the
sample inside a shared-memory dictionary. Subsequent workers (e.g., other DataLoader
workers on the same node) pull from that cache, reducing redundant reads.

.. note::

    Set ``persistent_workers=True`` on your DataLoader so workers stay alive;
    :class:`~aiaccel.torch.lightning.datamodules.SingleDataModule` enables this by
    default. Without persistent workers the shared-memory cache is lost when workers
    respawn.

.. note::
    On systems with small ``/dev/shm`` allocations, use
    :class:`~aiaccel.torch.datasets.FileCachedDataset` instead so cached samples live
    on disk (``cache_path``).

Writing datasets with ``HDF5Writer``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many HPC filesystems penalize workloads that touch thousands of small files (Lustre in
particular struggles with lots of metadata operations), so packaging training samples into a
single HDF5 file avoids pathological seeks. HDF5 also supports parallel I/O, allowing
you to read or write shards concurrently. To dump an entire dataset or inference pass, subclass
:class:`~aiaccel.torch.h5py.HDF5Writer` and define two hooks:

- :meth:`~aiaccel.torch.h5py.HDF5Writer.prepare_globals` returns the list of items to
  process (e.g., sample IDs) plus any context.
- :meth:`~aiaccel.torch.h5py.HDF5Writer.prepare_group` receives one item and returns a
  mapping ``{group_name: {dataset_name: np.ndarray}}`` describing how to populate the
  HDF5 file.

The writer takes care of serializing everything, and emits a ``.json`` index of created
groups. Set ``parallel=True`` when calling :meth:`~aiaccel.torch.h5py.HDF5Writer.write`
to leverage Parallel HDF5. A minimal writer that dumps embeddings might look like:

.. code-block:: python
    :caption: Custom HDF5Writer

    import numpy as np
    from aiaccel.torch.h5py import HDF5Writer

    class EmbeddingWriter(HDF5Writer[int, None]):
        def __init__(self, features):
            self.features = features

        def prepare_globals(self):
            return list(range(len(self.features))), None

        def prepare_group(self, idx, context):
            vec = self.features[idx]
            return {f"sample_{idx:06d}": {"embedding": vec}}

    writer = EmbeddingWriter(np.load("embeddings.npy"))
    writer.write(Path("artifacts/data.h5"))

Once persisted, you can stream the data back with
:class:`~aiaccel.torch.datasets.HDF5Dataset` or
:class:`~aiaccel.torch.datasets.RawHDF5Dataset`, which read the ``.h5`` / ``.json`` pair
and expose a PyTorch-compatible dataset. This pairing allows you to precompute heavy
features once and reuse them across training or inference jobs without re-running the
original pipeline.

.. note::

    Parallel HDF5 requires launching the script under MPI. The ``aiaccel-torch`` CLI can
    handle this for you via the MPI job mode:

    .. code-block:: bash

        aiaccel-job local mpi --n_procs=32 generate_hdf5.log -- \
            generate_hdf5.py

    Replace ``local`` with ``pbs`` / ``sge`` for cluster use. Inside ``generate_hdf5.py``
    call ``writer.write(..., parallel=True)`` so every MPI rank contributes to the same
    HDF5 file.

Further reading
---------------

- :doc:`../api_reference/torch` for the full API reference of datasets, pipelines,
  Lightning helpers, schedulers, and I/O utilities.
- :doc:`config` for composition patterns, resolvers, and CLI overrides used by torch
  configs.
- :doc:`job` for the job-launching interface that wraps the ``aiaccel-torch`` command.
