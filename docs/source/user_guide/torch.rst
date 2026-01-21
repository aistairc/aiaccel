##########################
 Training a PyTorch Model
##########################

*****************
 Getting Started
*****************

Aiaccel-based training is a wrapper of PyTorch Lightning, which can be executed as
follows:

.. code-block:: bash

    python -m aiaccel.torch.apps.train config.yaml

The config file `config.yaml` typically consists of `trainer`, `datamodule`, and `task`
as follows:

.. code-block:: yaml
    :caption: config.yaml
    :linenos:

     _base_: ${resolve_pkg_path:aiaccel.torch.apps.config}/train_base.yaml

     trainer:
       max_epochs: 10

       callbacks:
         - _target_: lightning.pytorch.callbacks.ModelCheckpoint
           filename: "{epoch:04d}"
           save_last: True
           save_top_k: -1

     datamodule:
       _target_: aiaccel.torch.lightning.datamodules.SingleDataModule

       train_dataset_fn:
         _partial_: True
         _target_: torchvision.datasets.MNIST

         root: "./dataset"
         train: True
         download: True

         transform:
           _target_: torchvision.transforms.Compose
           transforms:
             - _target_: torchvision.transforms.Resize
               size: [[256, 256]]
             - _target_: torchvision.transforms.Grayscale
               num_output_channels: 3
             - _target_: torchvision.transforms.ToTensor
             - _target_: torchvision.transforms.Normalize
               mean: [0.5]
               std: [0.5]

       val_dataset_fn:
         _partial_: True
         _inherit_: ${datamodule.train_dataset_fn}

         train: False

       batch_size: 128
       wrap_scatter_dataset: False

     task:
       _target_: my_task.MyTask
       num_classes: 10

       model:
         _target_: torchvision.models.resnet50
         weights:
           _target_: hydra.utils.get_object
           path: torchvision.models.ResNet50_Weights.DEFAULT

       optimizer_config:
         _target_: aiaccel.torch.lightning.OptimizerConfig
         optimizer_generator:
           _partial_: True
           _target_: torch.optim.Adam
           lr: 1.e-4

************************
 Multiple LR Schedulers
************************

``OptimizerConfig`` now accepts multiple schedulers so you can mix intervals, for example a step-wise warmup together
with an epoch-wise schedule like ``ReduceLROnPlateau``.

.. code-block:: yaml
    :caption: Mixing warmup (step) and plateau (epoch)
    :linenos:

     optimizer_config:
       _target_: aiaccel.torch.lightning.OptimizerConfig
       optimizer_generator:
         _partial_: True
         _target_: torch.optim.AdamW
         lr: 3.e-4

       schedulers:
         - _target_: aiaccel.torch.lightning.LRSchedulerConfig
           scheduler_generator:
             _partial_: True
             _target_: torch.optim.lr_scheduler.LinearLR
             start_factor: 0.1
             end_factor: 1.0
             total_iters: 500
           interval: step

         - _target_: aiaccel.torch.lightning.LRSchedulerConfig
           scheduler_generator:
             _partial_: True
             _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
             reduce_on_plateau: True
             factor: 0.5
             patience: 2
           interval: epoch
           monitor: validation/loss

When using ``ReduceLROnPlateau`` remember to log the metric specified in ``monitor``.

**********************
 Distributed Training
**********************

WIP...

*****************
 Other Utilities
*****************

Other utilities are listed in :doc:`API Reference <../api_reference/torch>`.
