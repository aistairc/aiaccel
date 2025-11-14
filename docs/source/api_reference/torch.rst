PyTorch/Lightning Toolkit
=========================

Datasets
--------

.. currentmodule:: aiaccel.torch.datasets

.. autosummary::
    :toctree: generated/

    CachedDataset
    FileCachedDataset
    HDF5Dataset
    RawHDF5Dataset
    scatter_dataset

Functional
----------

.. currentmodule:: aiaccel.torch.functional

.. autosummary::
    :toctree: generated/

    linear_sum_assignment

Learning Rate Schedulers
------------------------

.. currentmodule:: aiaccel.torch.lr_schedulers

.. autosummary::
    :toctree: generated/

    SequentialLR

Inference Pipeline Helpers
--------------------------

.. currentmodule:: aiaccel.torch.pipelines

.. autosummary::
    :toctree: generated/

    BasePipeline
    reorder_fields


Lightning Utilities
-------------------

.. currentmodule:: aiaccel.torch.lightning

.. autosummary::
    :toctree: generated/

    OptimizerLightningModule
    OptimizerConfig
    build_param_groups
    load_checkpoint
    ABCIEnvironment

Lightning Datamodules
^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: aiaccel.torch.lightning.datamodules

.. autosummary::
    :toctree: generated/

    SingleDataModule

H5py Utilities
--------------

.. currentmodule:: aiaccel.torch.h5py

.. autosummary::
    :toctree: generated/

    HDF5Writer