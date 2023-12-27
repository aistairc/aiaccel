from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import lightning
import numpy as np
import torch
import torch.utils
from omegaconf import DictConfig, OmegaConf
from torch import nn

from aiaccel.nas.batch_dependent_lr_scheduler import create_batch_dependent_lr_scheduler
from aiaccel.nas.create_optimizer import create_optimizer
from aiaccel.nas.utils import utils
from aiaccel.nas.utils.logger import (
    create_architecture_search_logger,
    create_architecture_search_report,
    create_supernet_train_logger,
    create_supernet_train_report,
)
from aiaccel.nas.utils.utils import make_observed_values2

if TYPE_CHECKING:
    from pathlib import Path

    from nas.module.nas_module import NASModule
    from torch.nn import Module
    from torch.optim import Optimizer


def _create_batch_dependent_lr_scheduler(
    optimizer: Optimizer,
    parameter_config: DictConfig,
    num_epochs: int,
    num_batches: int,
):
    hyperparameters = OmegaConf.to_container(parameter_config, resolve=True)
    return create_batch_dependent_lr_scheduler(optimizer, hyperparameters, num_epochs, num_batches)


def _create_optimizer(
    nn_module: Module,
    parameter_config: DictConfig,
):
    hyperparameters = OmegaConf.to_container(parameter_config, resolve=True)
    return create_optimizer(nn_module, hyperparameters)


class MnasnetTrainModel(lightning.LightningModule):
    def __init__(
        self,
        dataloader,
        nn_model: NASModule,
        search_space_config: tuple[int, dict[str, Any] | None],
        # hyperparameters: dict[str, ParameterType],
        parameter_config: DictConfig,
        categories,
        asng,
        structure_info,
        los_func: nn.CrossEntropyLoss,
        log_dir: Path,
        parent_logger_name: str,
        nas_config: DictConfig,
        num_train_data_sampler: int,
    ):
        super().__init__()
        _dims = dataloader.get_dims()
        self.channels = _dims[0]
        self.width = _dims[1]
        self.height = _dims[2]
        self.num_classes = dataloader.get_num_classes()
        # device_ids = (
        #    None if nas_config.environment.device_ids is None else list(map(int, nas_config.environment.device_ids))
        # )
        # self.model = nn.DataParallel(nn_model, device_ids=device_ids)
        self.model = nn_model
        self._search_space_config = search_space_config
        self._parameter_config = parameter_config
        self.categories = categories
        self.asng = asng
        self.num_train_data = dataloader.get_num_supernet_train_data()
        self.num_train_data_sampler = num_train_data_sampler
        self.los_func = los_func
        self.lam = self._parameter_config.lam
        self.log_dir = log_dir
        self.parent_logger_name = parent_logger_name
        self.nas_config = nas_config

        self.structure_info = structure_info
        self.automatic_optimization = False  # Activates manual optimization in training_step method.

        self.train_loss = 0.0
        self.train_acc1 = 0.0
        self.train_acc5 = 0.0

        self.start = None
        self.custom_logger = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        self.scheduler.batch_step()
        optimizer = self.optimizers()
        inputs, target = batch
        loss_sum = 0.0
        optimizer.zero_grad()

        for _ in range(self.lam):
            observed_values_one_hot = self.asng.sampling()
            observed_values = np.argmax(observed_values_one_hot, axis=1)
            observed_values_new = make_observed_values2(observed_values, self._search_space_config, self.categories)
            self.structure_info.update_values(observed_values_new)
            # self.model.module.select_active_op(self.structure_info)
            self.model.select_active_op(self.structure_info)

            output = self.model(inputs)
            loss = self.los_func(output, target)
            loss_sum = loss_sum + loss / self.lam

            self.train_loss += loss.item() * len(inputs) / (self.num_train_data_sampler * self.lam)
            self.train_acc1 += (
                utils.accuracy(output, target, topk=(1,))[0].item()
                * len(inputs)
                / (self.num_train_data_sampler * self.lam)
            )
            self.train_acc5 += (
                utils.accuracy(output, target, topk=(5,))[0].item()
                * len(inputs)
                / (self.num_train_data_sampler * self.lam)
            )

            del loss, output

        loss_sum = np.average(self.all_gather(loss_sum).numpy())
        self.manual_backward(loss_sum)
        del loss_sum, inputs, target
        optimizer.step()
        self.scheduler.step()

    def configure_optimizers(self):
        return _create_optimizer(self.model, self._parameter_config)

    def on_train_start(self):
        self.custom_logger = create_supernet_train_logger(
            self.log_dir,
            self.parent_logger_name + ".supernet_train_result",
        )
        self.custom_logger.info(f"The number of training images: {self.num_train_data_sampler}")
        self.custom_logger.info(f"The number of batches: {self.num_train_data}")
        self.start = time.time()
        optimizer = self.optimizers()
        self.scheduler = _create_batch_dependent_lr_scheduler(
            optimizer,
            self._parameter_config,
            self.nas_config.nas.num_epochs_supernet_train,
            self.num_train_data,
        )

    def on_train_epoch_start(self):
        self.model.train()
        self.train_loss = 0.0
        self.train_acc1 = 0.0
        self.train_acc5 = 0.0

    def on_train_epoch_end(self):
        elapsed_time = time.time() - self.start
        optimizer = self.optimizers()

        self.custom_logger.info(
            msg="",
            extra=create_supernet_train_report(
                epoch=self.current_epoch,
                elapsed_time=elapsed_time,
                train_loss=self.train_loss,
                top_1_train_acc=self.train_acc1,
                top_5_train_acc=self.train_acc5,
                learning_rate=optimizer.param_groups[0]["lr"],
            ),
        )
        self.log("train_acc1", self.train_acc1, prog_bar=True, on_epoch=True)


class MnasnetSearchModel(lightning.LightningModule):
    def __init__(
        self,
        dataloader,
        nn_model: NASModule,
        search_space_config: tuple[int, dict[str, Any] | None],
        parameter_config: DictConfig,
        categories,
        asng,
        structure_info,
        los_func: nn.CrossEntropyLoss,
        log_dir: Path,
        parent_logger_name: str,
        num_train_data_sampler: int,
    ):
        super().__init__()
        _dims = dataloader.get_dims()
        self.channels = _dims[0]
        self.width = _dims[1]
        self.height = _dims[2]
        self.num_classes = dataloader.get_num_classes()
        self.model = nn_model
        self._search_space_config = search_space_config
        self._parameter_config = parameter_config
        self.categories = categories
        self.asng = asng
        self.num_train_data = dataloader.get_num_architecture_search_data()
        self.num_train_data_sampler = num_train_data_sampler
        self.los_func = los_func
        self.lam = self._parameter_config.lam
        self.log_dir = log_dir
        self.parent_logger_name = parent_logger_name

        self.structure_info = structure_info
        self.automatic_optimization = False  # Activates manual optimization in training_step method.

        self.scheduler = None

        self.valid_loss = 0.0
        self.valid_acc1 = 0.0
        self.valid_acc5 = 0.0

        self.start = None
        self.custom_logger = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        observed_values_one_hot_list = []
        losses = []

        with torch.no_grad():
            for _ in range(self.lam):
                observed_values_one_hot = self.asng.sampling()
                observed_values = np.argmax(observed_values_one_hot, axis=1)
                observed_values_new = utils.make_observed_values2(
                    observed_values,
                    self._search_space_config,
                    self.categories,
                )
                self.structure_info.update_values(observed_values_new)
                # self.model.module.select_active_op(self.structure_info)
                self.model.select_active_op(self.structure_info)
                h_valid = self.model(inputs)
                loss = self.los_func(h_valid, target)
                self.valid_loss += loss.item() * len(inputs) / (self.num_train_data_sampler * self.lam)
                self.valid_acc1 += (
                    utils.accuracy(h_valid, target, topk=(1,))[0].item()
                    * len(inputs)
                    / (self.num_train_data_sampler * self.lam)
                )
                self.valid_acc5 += (
                    utils.accuracy(h_valid, target, topk=(5,))[0].item()
                    * len(inputs)
                    / (self.num_train_data_sampler * self.lam)
                )
                observed_values_one_hot_list.append(observed_values_one_hot)
                losses.append(loss.item())
                del h_valid, loss

        losses, observed_values_one_hot_list = np.array(losses), np.array(observed_values_one_hot_list)
        losses = np.average(self.all_gather(losses).numpy(), axis=0)
        observed_values_one_hot_list = np.average(self.all_gather(observed_values_one_hot_list).numpy(), axis=0)
        self.asng.update(observed_values_one_hot_list, losses)
        del inputs, target

    def configure_optimizers(self):
        return _create_optimizer(self.model, self._parameter_config)

    def on_train_start(self):
        self.custom_logger = create_architecture_search_logger(
            self.log_dir,
            self.parent_logger_name + ".architecture_search_result",
        )
        self.start = time.time()

    def on_train_epoch_start(self):
        self.valid_loss = 0.0
        self.valid_acc1 = 0.0
        self.valid_acc5 = 0.0

    def on_train_epoch_end(self):
        elapsed_time = time.time() - self.start
        convergence = self.asng.p_model.theta.max(axis=1).mean()
        self.custom_logger.info(
            msg="",
            extra=create_architecture_search_report(
                epoch=self.current_epoch,
                elapsed_time=elapsed_time,
                valid_loss=self.valid_loss,
                top_1_valid_acc=self.valid_acc1,
                top_5_valid_acc=self.valid_acc5,
                convergence=convergence,
            ),
        )


class MnasnetRerainModel(lightning.LightningModule):
    pass
