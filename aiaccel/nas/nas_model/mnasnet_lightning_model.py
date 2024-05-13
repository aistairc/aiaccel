from __future__ import annotations

import copy
import gc
import math
import pickle
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lightning
import numpy as np
import torch
import torch.utils
import torchvision
from omegaconf import DictConfig, OmegaConf
from ptflops import get_model_complexity_info
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader

from aiaccel.nas.asng.categorical_asng import CategoricalASNG
from aiaccel.nas.batch_dependent_lr_scheduler import create_batch_dependent_lr_scheduler
from aiaccel.nas.create_optimizer import create_optimizer
from aiaccel.nas.nas_model.proxyless_model import MnasNetSearchSpace
from aiaccel.nas.utils import utils
from aiaccel.nas.utils.logger import (
    create_architecture_search_logger,
    create_architecture_search_report,
    create_logger,
    create_retrain_report,
    create_supernet_train_logger,
    create_supernet_train_report,
    create_train_and_search_logger,
    create_train_and_search_report,
)
from aiaccel.nas.utils.utils import (
    _data_transforms_cifar,
    _data_transforms_imagenet,
    create_config_by_yaml,
    cross_entropy_with_label_smoothing,
    get_device,
    get_params2,
    get_random_dataset_directory,
    get_search_space_config,
    load_imagenet_dataset,
    make_categories2,
    make_observed_values2,
)

if TYPE_CHECKING:
    from os import PathLike
    from typing import Callable

    from nas.module.nas_module import NASModule
    from torch.nn import Module
    from torch.optim import Optimizer

    from aiaccel.nas.mnas_structure_info import MnasNetStructureInfo


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


def _adjust_learning_rate(
    optimizer,
    epoch,
    epochs,
    batch_idx,
    len_train_loader,
    base_lr,
    adjust_type="cosine",
    warmup_epochs=5,
    h_size=3,
):
    lr_adj = 0

    if epoch < warmup_epochs:
        epoch += float(batch_idx + 1) / len_train_loader
        lr_adj = 1.0 / h_size * (epoch * (h_size - 1) / warmup_epochs + 1)
    elif adjust_type == "linear":
        if epoch < 30:
            lr_adj = 1.0
        elif epoch < 60:
            lr_adj = 1e-1
        elif epoch < 90:
            lr_adj = 1e-2
        else:
            lr_adj = 1e-3
    elif adjust_type == "cosine":
        run_epochs = epoch - warmup_epochs
        total_epochs = epochs - warmup_epochs
        t_cur = float(run_epochs * len_train_loader) + batch_idx
        t_total = float(total_epochs * len_train_loader)

        lr_adj = 0.5 * (1 + math.cos(math.pi * t_cur / t_total))

    for param_group in optimizer.param_groups:
        param_group["lr"] = base_lr * lr_adj


def kl_divergence(p, q):
    return sum(p[i] * np.log(p[i] / q[i]) for i in range(len(p)))


def get_params(nn_model):
    return nn.utils.parameters_to_vector(nn_model.parameters()).to("cpu").detach().numpy().copy()


def get_params_grad(nn_model):
    nun = 0
    for param in nn_model.parameters():
        if param.grad is not None:
            if nun == 0:
                all_params_grads = param.grad.view(-1).to("cpu").detach().numpy().copy()
                nun += 1
            else:
                all_params_grads = np.append(all_params_grads, param.grad.view(-1).to("cpu").detach().numpy().copy())
        elif nun == 0:
            all_params_grads = param.new_zeros(param.size()).view(-1).to("cpu").detach().numpy().copy()
            nun += 1
        else:
            all_params_grads = np.append(
                all_params_grads,
                param.new_zeros(param.size()).view(-1).to("cpu").detach().numpy().copy(),
            )

    return all_params_grads


def _get_best_result_directory(log_root_dir: PathLike) -> Path:
    """Gets directory name of the best architecture search log.

    Args:
        log_root_dir (PathLike): Path to root directory of log.

    Raises:
        FileNotFoundError: Causes when no log file is found in the specified
            log_root_dir.

    Returns:
        Path: Absolute path to directory for best architecture search log.
    """
    if result_files := log_root_dir.glob("**/result.pkl"):
        current_best_accuracy = -float("inf")
        current_best_result_directory = ""

        for log_dir in result_files:
            with (Path(log_dir).resolve()).open("rb") as f:
                result = pickle.load(f)

            if (
                isinstance(result, dict)
                and "validation_accuracy" in result
                and result["validation_accuracy"] > current_best_accuracy
            ):
                current_best_result_directory = log_dir
                current_best_accuracy = result["validation_accuracy"]

        return Path(current_best_result_directory).resolve()
    raise FileNotFoundError(log_root_dir)


# class MnasnetTrainer:
#    _nas_config: DictConfig
#    _parameter_config: DictConfig
#    _search_space_config: dict


def _test(
    nn_model: nn.DataParallel,
    loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    dataloader: DataLoader,
) -> tuple[float, float, float]:
    """Function for deterministic evaluation.

    The most likely architecture is used.

    Args:
        nn_model (nn.DataParallel): _description_
        loss_func (Callable[[torch.Tensor, torch.Tensor], torch.Tensor]): _description_
        dataloader (DataLoader): _description_

    Returns:
        tuple[float, float, float]: _description_

    Note::

        test_loader = DataLoader(test_data, batch_size, shuffle=False, num_workers=12)
    """
    device = get_device()
    num_data = len(dataloader.sampler)

    nn_model.eval()

    loss_avg = 0.0
    acc1_avg = 0.0
    acc5_avg = 0.0

    with torch.no_grad():
        for X, t in dataloader:
            X, t = X.to(device), t.to(device)

            output = nn_model(X)
            loss_avg += loss_func(output, t).item() * len(X) / num_data

            acc1_avg += utils.accuracy(output, t, topk=(1,))[0].item() * len(X) / num_data
            acc5_avg += utils.accuracy(output, t, topk=(5,))[0].item() * len(X) / num_data

    return loss_avg, acc1_avg, acc5_avg


class MnasnetTrainModel(lightning.LightningModule):
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
        nas_config: DictConfig,
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

        if self.nas_config.environment.gpus is not None and self.nas_config.environment.gpus > 1:
            loss_sum = np.average(self.all_gather(loss_sum).numpy(force=True))
            self.manual_backward(float(loss_sum))
        else:
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
        nas_config: DictConfig,
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
        self.nas_config = nas_config

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
        if (
            self.nas_config is not None
            and self.nas_config.environment.gpus is not None
            and self.nas_config.environment.gpus > 1
        ):
            losses = np.average(self.all_gather(losses).numpy(force=True), axis=0)
            observed_values_one_hot_list = np.average(
                self.all_gather(observed_values_one_hot_list).numpy(force=True),
                axis=0,
            )
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


class MnasnetTrainSearchModel(lightning.LightningModule):
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
        nas_config: DictConfig,
        num_train_data_sampler: int,
        valid_dataloader: DataLoader,
        test_dataloader: DataLoader,
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
        self.num_train_data = dataloader.get_num_supernet_train_data()
        self.num_train_data_sampler = num_train_data_sampler
        self.los_func = los_func
        self.lam = self._parameter_config.lam
        self.log_dir = log_dir
        self.parent_logger_name = parent_logger_name
        self.nas_config = nas_config

        self.structure_info = structure_info
        self.automatic_optimization = False  # Activates manual optimization in training_step method.

        self.scheduler = None

        self.train_loss = 0.0
        self.train_acc1 = 0.0
        self.train_acc5 = 0.0
        self.valid_loss = 0.0
        self.valid_acc1 = 0.0
        self.valid_acc5 = 0.0
        self.valid_dataloader = valid_dataloader
        self.valid_iter = valid_dataloader.__iter__()
        self.num_valid_data_sampler = len(self.valid_dataloader.sampler)
        self.test_dataloader = test_dataloader

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

        if self.nas_config.environment.gpus is not None and self.nas_config.environment.gpus > 1:
            loss_sum = np.average(self.all_gather(loss_sum).numpy(force=True))
            self.manual_backward(float(loss_sum))
        else:
            self.manual_backward(loss_sum)
        del loss_sum, inputs, target
        optimizer.step()
        self.scheduler.step()

        observed_values_one_hot_list = []
        losses = []

        if self.current_epoch >= 20 and batch_idx < len(self.val_dataloader):
            x_valid, t_valid = self.valid_iter.next()
            x_valid, t_valid = x_valid.to(self.device), t_valid.to(self.device)

            with torch.no_grad():
                for _ in range(self.lam):
                    observed_values_one_hot = self.asng.sampling()
                    observed_values = np.argmax(observed_values_one_hot, axis=1)
                    observed_values_new = make_observed_values2(
                        observed_values,
                        self._search_space_config,
                        self.categories,
                    )
                    self.structure_info.update_values(observed_values_new)
                    self.model.select_active_op(self.structure_info)
                    h_valid = self.model(x_valid)
                    loss = self.los_func(h_valid, t_valid)

                    self.valid_loss += loss.item() * len(x_valid) / (self.num_train_data_sampler * self.lam)
                    self.valid_acc1 += (
                        utils.accuracy(h_valid, t_valid, topk=(1,))[0].item()
                        * len(x_valid)
                        / (self.num_train_data_sampler * self.lam)
                    )
                    self.valid_acc5 += (
                        utils.accuracy(h_valid, t_valid, topk=(5,))[0].item()
                        * len(x_valid)
                        / (self.num_train_data_sampler * self.lam)
                    )
                    observed_values_one_hot_list.append(observed_values_one_hot)
                    losses.append(loss.item())
                    del h_valid, loss

            losses, observed_values_one_hot_list = np.array(losses), np.array(observed_values_one_hot_list)
            if (
                self.nas_config is not None
                and self.nas_config.environment.gpus is not None
                and self.nas_config.environment.gpus > 1
            ):
                losses = np.average(self.all_gather(losses).numpy(force=True), axis=0)
                observed_values_one_hot_list = np.average(
                    self.all_gather(observed_values_one_hot_list).numpy(force=True),
                    axis=0,
                )
            self.asng.update(observed_values_one_hot_list, losses)
            del x_valid, t_valid

    def configure_optimizers(self):
        return _create_optimizer(self.model, self._parameter_config)

    def on_train_start(self):
        self.custom_logger = create_train_and_search_logger(
            self.log_dir,
            self.parent_logger_name + ".train_and_search_result",
        )
        self.custom_logger.info(f"The number of training images: {self.num_train_data_sampler}")
        self.custom_logger.info(f"The number of validation images: {self.num_valid_data_sampler}")
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
        self.valid_loss = 0.0
        self.valid_acc1 = 0.0
        self.valid_acc5 = 0.0

    def on_train_epoch_end(self):
        test_loss = test_acc1 = test_acc5 = np.nan
        if self.test_dataloader is not None:
            mle_one_hot = self.asng.p_model.mle()
            mle = np.argmax(mle_one_hot, axis=1)
            mle_new = utils.make_observed_values2(mle, self._search_space_config, self.categories)
            self.structure_info.update_values(mle_new)
            self.model.module.select_active_op(self.structure_info)
            # if minus_test_time:
            test_start = time.time()
            test_loss, test_acc1, test_acc5, threshold = _test(self.model, self.loss_func, self.test_dataloader)
            self.start += time.time() - test_start

        elapsed_time = time.time() - self.start

        if self.current_epoch == 0:
            p_old = np.array(self.asng.p_model.log()).astype("float")
            params_old = get_params(self.model)
            upper_bound = np.nan
            tprime = np.nan
            residual = np.nan
        else:
            p_new = np.array(self.asng.p_model.log()).astype("float")
            kl = kl_divergence(p_old, p_new)
            p_old = copy.deepcopy(p_new)
            upper_b = 2
            params_new = get_params(self.model)
            all_params_grads = get_params_grad(self.model)
            tprime = np.dot(all_params_grads, (params_new - params_old))
            residual = upper_b * np.linalg.norm(params_new - params_old)
            upper_bound = upper_b * np.sqrt(0.5 * kl) + tprime + residual
            params_old = copy.deepcopy(params_new)

        convergence = self.asng.p_model.theta.max(axis=1).mean()
        optimizer = self.optimizers()

        self.custom_logger.info(
            msg="",
            extra=create_train_and_search_report(
                epoch=self.current_epoch,
                elapsed_time=elapsed_time,
                train_loss=self.train_loss,
                top_1_train_acc=self.train_acc1,
                top_5_train_acc=self.train_acc5,
                valid_loss=self.valid_loss,
                top_1_valid_acc=self.valid_acc1,
                top_5_valid_acc=self.valid_acc5,
                test_loss=test_loss,
                top_1_test_acc=test_acc1,
                top_5_test_acc=test_acc5,
                convergence=convergence,
                learning_rate=optimizer.param_groups[0]["lr"],
                upper_bound=upper_bound,
                threshold=threshold,
            ),
        )
        self.log("train_acc1", self.train_acc1, prog_bar=True, on_epoch=True)


class MnasnetRerainModel(lightning.LightningModule):
    def __init__(
        self,
        search_space_config: tuple[int, dict[str, Any] | None],
        parameter_config: DictConfig,
        nas_config: DictConfig,
        log_dir: Path,
        parent_logger_name: str = "root",
    ):
        super().__init__()
        self._search_space_config = search_space_config
        self._parameter_config = parameter_config
        self._nas_config = nas_config
        self.log_dir = log_dir
        self.parent_logger_name = parent_logger_name

        self.train_loss = 0.0
        self.train_acc1 = 0.0
        self.train_acc5 = 0.0
        self.test_loss = 0.0
        self.test_acc1 = 0.0
        self.test_acc5 = 0.0

        self.start = None
        self.custom_logger = None
        self.log_dir = log_dir

        # retrain_environment = self._nas_config.environment
        # result_dir = Path(retrain_environment.result_dir).resolve()
        # num_workers = int(retrain_environment.num_workers)
        num_workers = int(self._nas_config.environment.num_workers)
        retrain_dataset = self._nas_config.dataset
        train_data_path = Path(retrain_dataset.retrain.train)
        test_data_path = Path(retrain_dataset.retrain.test) if retrain_dataset.retrain.test else None
        num_classes = int(retrain_dataset.num_search_classes)

        retrain_hyperparameters = self._nas_config.retrain.hyperparameters
        batch_size = int(retrain_hyperparameters.batch_size)
        self.n_epochs = int(retrain_hyperparameters.num_epochs)
        self.base_lr = float(retrain_hyperparameters.base_lr)
        momentum = float(retrain_hyperparameters.momentum)
        weight_decay = float(retrain_hyperparameters.weight_decay)

        self.log_dir = self.log_dir / f"retrain-{datetime.now().strftime('%d-%m-%Y-%H-%M-%S-%f')}"
        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True)

        logger = create_logger(self.log_dir, "root.retrain")
        logger.info(str(self.log_dir))

        try:
            best_result_directory = _get_best_result_directory(Path(self._nas_config.environment.result_dir).resolve())
        except BaseException as exc:
            logger.exception("Could not find best result directory.")
            raise BaseException from exc

        logger.info(f"Best result of architecture search: {best_result_directory.parent}")

        shutil.copytree(best_result_directory.parent, str(self.log_dir / "architecture_search_result"))

        with best_result_directory.open("rb") as f:
            result = pickle.load(f)

        trained_theta: list[np.ndarray[Any, np.dtype[float]]] = result["trained_theta"]
        structure_info: MnasNetStructureInfo = result["structure_info"]

        hyperparameters: dict[str, Any] = result["hyperparameters"]
        alpha = hyperparameters["alpha"]
        epsilon = hyperparameters["epsilon"]

        logger.info("Retraining")

        _search_space_config_path = get_search_space_config(
            self._nas_config.nas.search_space,
            self._nas_config.dataset.name,
        )
        self._search_space_config = create_config_by_yaml(
            self._nas_config.nas.search_space_config_path + str(_search_space_config_path),
        )

        if retrain_dataset.name == "imagenet":
            directory_list, label_map = get_random_dataset_directory(train_data_path, num_classes=num_classes)
            train_transform, test_transform = _data_transforms_imagenet()
            train_dataset = load_imagenet_dataset(
                train_data_path,
                directory_list,
                label_map,
                transform=train_transform,
                train=True,
            )
        elif retrain_dataset.name == "cifar10":
            train_transform, test_transform = _data_transforms_cifar()
            train_dataset = torchvision.datasets.CIFAR10(
                train_data_path,
                train=True,
                transform=train_transform,
                download=True,
            )
        elif retrain_dataset.name == "cifar100":
            train_transform, test_transform = _data_transforms_cifar()
            train_dataset = torchvision.datasets.CIFAR100(
                train_data_path,
                train=True,
                transform=train_transform,
                download=True,
            )
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False,
        )
        self.num_train_batches = len(self.train_dataloader)
        self.num_train_data = len(self.train_dataloader.sampler)

        if test_data_path is not None and test_data_path.exists() and retrain_dataset.name == "imagenet":
            valid_dataset = load_imagenet_dataset(
                test_data_path,
                directory_list,
                label_map,
                transform=test_transform,
                train=False,
            )
            self.valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=False, num_workers=num_workers)
        elif test_data_path is not None and test_data_path.exists() and retrain_dataset.name == "cifar10":
            valid_dataset = torchvision.datasets.CIFAR10(
                test_data_path,
                train=False,
                transform=test_transform,
                download=True,
            )
            self.valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=False, num_workers=num_workers)
        elif test_data_path is not None and test_data_path.exists() and retrain_dataset.name == "cifar100":
            valid_dataset = torchvision.datasets.CIFAR100(
                test_data_path,
                train=False,
                transform=test_transform,
                download=True,
            )
            self.valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=False, num_workers=num_workers)
        else:
            self.valid_dataloader = None

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = get_device()

        self.model = MnasNetSearchSpace("MnasNetSearchSpace")
        self.model.build(self._search_space_config)
        category_dic = self.model.enumerate_categorical_variables()
        categories = make_categories2(category_dic, self._search_space_config)
        params = get_params2(self.model, structure_info, self._search_space_config, categories)
        init_delta = 1.0 / sum(categories)

        self.asng = CategoricalASNG(
            categories,
            params,
            alpha=alpha,
            eps=epsilon,
            init_delta=init_delta,
            init_theta=trained_theta,
        )

        mle_one_hot = self.asng.p_model.mle()
        mle = np.argmax(mle_one_hot, axis=1)
        mle_new = make_observed_values2(mle, self._search_space_config, categories)

        structure_info.update_values(mle_new)
        self.model.select_active_op(structure_info)
        self.model.fix_arc()

        logger.info(f"The numbers of parameters: {self.model.get_param_num_list()}")

        p = 0
        for param in self.model.parameters():
            p += param.numel()

        macs, params_count = get_model_complexity_info(
            self.model,
            (3, 224, 224),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )
        logger.info(f"flops: {macs / 1e6}")

        with (self.log_dir / "description.txt").open("a") as o:
            o.write("flops(after search)_" + str(epsilon) + ": %fM\n" % (macs / 1e6))

        if device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

        self.optimizer = SGD(
            self.model.parameters(),
            lr=self.base_lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        self.loss_func = cross_entropy_with_label_smoothing

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()
        _adjust_learning_rate(
            optimizer,
            self.current_epoch,
            self.n_epochs,
            batch_idx,
            self.num_train_batches,
            self.base_lr,
        )
        inputs, target = batch
        optimizer.zero_grad()
        output = self.model(inputs)
        loss = self.loss_func(output, target)

        if math.isnan(loss.item()):
            raise (optimizer.param_groups[0]["lr"])

        loss.backward()
        optimizer.step()
        self.train_loss += loss.item() * len(inputs) / self.num_train_data
        self.train_acc1 += utils.accuracy(output, target, topk=(1,))[0].item() * len(inputs) / self.num_train_data
        self.train_acc5 += utils.accuracy(output, target, topk=(5,))[0].item() * len(inputs) / self.num_train_data

    def train_dataloader(self):
        return self._train_dataloader

    def configure_optimizers(self):
        return self.optimizer

    # def configure_optimizers(self):
    #    return _create_optimizer(self.model, self._parameter_config)

    def on_train_start(self):
        self.custom_logger = create_supernet_train_logger(
            self.log_dir,
            self.parent_logger_name + ".retrain_result",
        )
        self.start = time.time()

    def on_train_epoch_start(self):
        self.model.train()
        self.train_loss = 0.0
        self.train_acc1 = 0.0
        self.train_acc5 = 0.0

    def on_train_epoch_end(self):
        self.test_loss = self.test_acc1 = self.test_acc5 = np.nan
        optimizer = self.optimizers()

        if self.valid_dataloader is not None:
            self.test_loss, self.test_acc1, self.test_acc5 = _test(self.model, self.loss_func, self.valid_dataloader)

        elapsed_time = time.time() - self.start
        convergence = self.asng.p_model.theta.max(axis=1).mean()

        self.custom_logger.info(
            msg="",
            extra=create_retrain_report(
                epoch=self.current_epoch,
                elapsed_time=elapsed_time,
                train_loss=self.train_loss,
                top_1_train_acc=self.train_acc1,
                top_5_train_acc=self.train_acc5,
                test_loss=self.test_loss,
                top_1_test_acc=self.test_acc1,
                top_5_test_acc=self.test_acc5,
                convergence=convergence,
                learning_rate=optimizer.param_groups[0]["lr"],
            ),
        )
