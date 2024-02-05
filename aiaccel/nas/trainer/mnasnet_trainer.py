from __future__ import annotations

import contextlib
import hashlib
import pickle
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logging import Logger

    from omegaconf import DictConfig
    from torch.utils.data import DataLoader

    from aiaccel.nas.data_module.nas_dataloader import NAS1shotDataLoader
import lightning
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import nn

from aiaccel.nas.asng.categorical_asng import CategoricalASNG
from aiaccel.nas.data_module.cifar10_data_module import Cifar10SubsetRandomSamplingDataLoader
from aiaccel.nas.mnas_structure_info import MnasNetStructureInfo
from aiaccel.nas.nas_model.mnasnet_lightning_model import MnasnetRerainModel, MnasnetSearchModel, MnasnetTrainModel
from aiaccel.nas.nas_model.proxyless_model import MnasNetSearchSpace
from aiaccel.nas.utils.logger import create_logger
from aiaccel.nas.utils.utils import (
    create_config_by_yaml,
    get_params2,
    get_search_space_config,
    make_categories2,
    make_observed_values2,
)


class MnasnetTrainer:
    _nas_config: DictConfig
    _parameter_config: DictConfig
    _search_space_config: dict
    _log_dir: Path
    _logger: Logger
    _los_func: nn.CrossEntropyLoss
    _dataloader: NAS1shotDataLoader
    _supernet_dataloader: DataLoader
    _architecture_search_datalaoder: DataLoader
    _architecture_search_asng: CategoricalASNG
    _categories: np.ndarray
    _structure_info: MnasNetStructureInfo
    _model: MnasNetSearchSpace
    _train_model: MnasnetTrainModel
    _search_model: MnasnetSearchModel
    _retrain_model: MnasnetRerainModel
    _supernet_trainer: lightning.Trainer
    _search_trainer: lightning.Trainer
    _valid_acc: float

    def __init__(self, nas_config: DictConfig, parameter_config: DictConfig):
        self._nas_config = nas_config
        self._parameter_config = parameter_config
        _search_space_config_path = get_search_space_config(nas_config.nas.search_space, nas_config.dataset.name)
        self._search_space_config = create_config_by_yaml(
            nas_config.nas.search_space_config_path + str(_search_space_config_path),
        )
        self._log_dir = None
        self._logger = None
        self._create_logger()
        lightning.seed_everything(nas_config.nas.seed, workers=True)
        self._los_func = nn.CrossEntropyLoss()
        self._dataloader = None
        self._supernet_dataloader = None
        self._architecture_search_datalaoder = None
        self._create_dataloader(_search_space_config_path)
        self._create_model(_search_space_config_path)

    def train(self):
        model_checkpoint_callback = ModelCheckpoint(
            dirpath=self._log_dir,
            filename="mnasnet_train_model-{epoch:02d}-{train_acc1:.2f}",
            monitor="train_acc1",
            save_top_k=1,
        )

        devices = "auto" if self._nas_config.environment.gpus is None else self._nas_config.environment.gpus
        self._supernet_trainer = lightning.Trainer(
            accelerator=self._nas_config.trainer.accelerator,
            callbacks=[model_checkpoint_callback],
            devices=devices,
            enable_model_summary=self._nas_config.trainer.enable_model_summary,
            enable_progress_bar=self._nas_config.trainer.enable_progress_bar,
            logger=self._nas_config.trainer.logger,
            max_epochs=self._nas_config.nas.num_epochs_supernet_train,
            strategy=self._nas_config.trainer.strategy,
        )
        self._logger.info("Start supernet train")
        self._supernet_trainer.fit(self._train_model, train_dataloaders=self._supernet_dataloader)
        self._logger.info("Supernet train finished")
        savepath = self._log_dir / "supernet_model.pt"
        self._logger.info(f"Save state dict of supernet: {savepath}")

        try:
            # torch.save(self._model.module.state_dict(), savepath)
            torch.save(self._model.state_dict(), savepath)
        except BaseException as exc:
            self._logger.exception("The train model could not be saved.")
            raise BaseException from exc

        self._logger.info("Supernet model saved")

    def search(self):
        self._logger.info("Start architecture search")
        model_checkpoint_callback = ModelCheckpoint(
            dirpath=self._log_dir,
            filename="mnasnet_search_model-{epoch:02d}-{train_acc1:.2f}",
            monitor="valid_acc1",
            save_top_k=1,
        )
        devices = "auto" if self._nas_config.environment.gpus is None else self._nas_config.environment.gpus
        self._search_trainer = lightning.Trainer(
            max_epochs=self._nas_config.nas.num_epochs_architecture_search,
            accelerator=self._nas_config.trainer.accelerator,
            strategy=self._nas_config.trainer.strategy,
            devices=devices,
            callbacks=[model_checkpoint_callback],
            enable_progress_bar=self._nas_config.trainer.enable_progress_bar,
            logger=self._nas_config.trainer.logger,
            enable_model_summary=self._nas_config.trainer.enable_model_summary,
        )
        self._search_trainer.fit(self._search_model, train_dataloaders=self._architecture_search_datalaoder)
        self._logger.info("Architecture search finished")

    def retrain(self):
        model_checkpoint_callback = ModelCheckpoint(
            dirpath=self._log_dir,
            filename="mnasnet_retrain_model-{epoch:02d}-{train_acc1:.2f}",
            monitor="valid_acc1",
            save_top_k=1,
        )
        devices = "auto" if self._nas_config.environment.gpus is None else self._nas_config.environment.gpus
        self._retrain_trainer = lightning.Trainer(
            max_epochs=self._nas_config.nas.num_epochs_retrain,
            accelerator=self._nas_config.trainer.accelerator,
            strategy=self._nas_config.trainer.strategy,
            devices=devices,
            callbacks=[model_checkpoint_callback],
            enable_progress_bar=self._nas_config.trainer.enable_progress_bar,
            logger=self._nas_config.trainer.logger,
            enable_model_summary=self._nas_config.trainer.enable_model_summary,
        )
        self._retrain_trainer.fit(self._retrain_model)
        self._logger.info("Retrain finished")

    def save(self):
        savepath = self._log_dir / "params_num_list.txt"
        self._logger.info(f"Save a list of the numbers of parameters: {savepath}")

        mle_one_hot = self._architecture_search_asng.p_model.mle()
        mle = np.argmax(mle_one_hot, axis=1)
        mle_new = make_observed_values2(mle, self._search_space_config, self._categories)
        self._structure_info.update_values(mle_new)
        self._model.select_active_op(self._structure_info)
        self._model.print_active_op(log_dir=self._log_dir)
        params_list = self._model.get_param_num_list()

        with savepath.open("a") as o:
            o.write(f"First conv layer: {params_list[0][0]}\n")

            for i in range(len(params_list) - 2):
                o.write(f"No.{i} block:\n")
                for j in params_list[i + 1]:
                    o.write(f"  - {j}\n")

            o.write(f"classifier: {params_list[-1][0]}")

        self._logger.info("List of the numbers of parameters saved")
        p = 0

        for param in params_list:
            p += sum(param)

        self._logger.info(f"The number of parameters: {p}")
        savepath = self._log_dir / "result.pkl"
        self._logger.info(f"Save result: {savepath}")
        trained_theta = self._architecture_search_asng.p_model.theta
        result = {
            "validation_accuracy": self.get_search_valid_acc(),
            "trained_theta": trained_theta,
            "structure_info": self._structure_info,
            "hyperparameters": self._parameter_config,
            "nas_config": self._nas_config,
        }

        with (self._log_dir / "result.pkl").open("wb") as f:
            pickle.dump(result, f)

        self._logger.info("Result saved")

    def get_search_valid_acc(self):
        return self._search_model.valid_acc1

    def get_retrain_test_acc(self):
        return self._retrain_model.test_acc1

    @property
    def is_global_zero(self):
        return self._search_trainer.is_global_zero

    def _create_logger(self):
        result_dir = Path(self._nas_config.environment.result_dir).resolve()

        if not result_dir.exists():
            result_dir.mkdir()

        self._log_dir = (
            Path(result_dir)
            / hashlib.sha3_512(
                f"{datetime.now().strftime('%d/%m/%Y, %H:%M:%S,%f')} {self._parameter_config}".encode(),
            ).hexdigest()
        )

        if not self._log_dir.exists():
            with contextlib.suppress(FileExistsError):
                self._log_dir.mkdir()

        self._logger = create_logger(self._log_dir, "root.search")
        self._logger.info(f"log directory: {self._log_dir}")

    def _create_dataloader(self, search_space_config_path: Path):
        self._logger.info(f"Load dataset: {self._nas_config.dataset.path}")
        if "cifar10" in str(search_space_config_path):
            self._dataloader = Cifar10SubsetRandomSamplingDataLoader(
                self._nas_config.dataset.path,
                self._parameter_config.batch_size_supernet_train,
                self._parameter_config.batch_size_architecture_search,
                self._nas_config.dataset.num_data_architecture_search,
                self._nas_config.environment.num_workers,
            )
            self._supernet_dataloader = self._dataloader.get_supernet_train_dataloader()
            self._architecture_search_datalaoder = self._dataloader.get_architecture_search_dataloader()
        else:
            raise ValueError(search_space_config_path)

        self._logger.info("Dataset loaded")
        self._logger.info("Create dataloaders")

    def _create_model(self, search_space_config_path: Path):
        if "proxyless" in str(search_space_config_path) or "mnasnet" in str(search_space_config_path):
            self._create_train_model()
            self._create_architecture_search_model()
            self._create_retrain_model()
        else:
            raise ValueError(search_space_config_path)
        self._logger.info(f"Output features: {self._model.classifier.out_features}")

    def _create_train_model(self):
        self._model = MnasNetSearchSpace("MnasNetSearchSpace")
        self._model.build(self._search_space_config)
        category_dic = self._model.enumerate_categorical_variables()
        self._structure_info = MnasNetStructureInfo(list(category_dic.keys()))
        self._categories = make_categories2(category_dic, self._search_space_config)
        init_delta = 1.0 / sum(self._categories)
        params = get_params2(self._model, self._structure_info, self._search_space_config, self._categories)
        supernet_asng = CategoricalASNG(
            self._categories,
            params,
            alpha=self._parameter_config.alpha,
            eps=0,
            init_delta=init_delta,
            init_theta=None,
        )
        self._logger.info(f"The total number of parameters for a categorical distribution: {sum(self._categories)}")
        self._logger.info(f"init_delta: {init_delta}")
        self._train_model = MnasnetTrainModel(
            self._dataloader,
            self._model,
            self._search_space_config,
            self._parameter_config,
            self._categories,
            supernet_asng,
            self._structure_info,
            self._los_func,
            self._log_dir,
            "root.search",
            self._nas_config,
            len(self._supernet_dataloader.sampler),
        )
        self._model = self._train_model.model

    def _create_architecture_search_model(self):
        params = get_params2(self._model, self._structure_info, self._search_space_config, self._categories)
        init_delta = 1.0 / sum(self._categories)
        self._architecture_search_asng = CategoricalASNG(
            self._categories,
            params,
            alpha=self._parameter_config.alpha,
            eps=self._parameter_config.eps,
            init_delta=init_delta,
            init_theta=None,
        )
        self._search_model = MnasnetSearchModel(
            self._dataloader,
            self._model,
            self._search_space_config,
            self._parameter_config,
            self._categories,
            self._architecture_search_asng,
            self._structure_info,
            self._los_func,
            self._log_dir,
            "root.search",
            self._nas_config,
            len(self._architecture_search_datalaoder.sampler),
        )

    def _create_retrain_model(self):
        self._retrain_model = MnasnetRerainModel(
            self._parameter_config,
            self._search_space_config,
            self._nas_config,
            self._log_dir,
            "root.retrain",
        )
