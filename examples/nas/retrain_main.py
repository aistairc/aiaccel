from __future__ import annotations

import gc
import glob
import pickle
import shutil
from argparse import ArgumentParser
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from nas.asng.categorical_asng import CategoricalASNG
from nas.mnas_structure_info import MnasNetStructureInfo
from nas.nas_model.proxyless_model import MnasNetSearchSpace
from nas.train import train
from nas.utils.logger import create_logger
from nas.utils.utils import (
    _data_transforms_cifar,
    _data_transforms_imagenet,
    create_config_by_yaml,
    cross_entropy_with_label_smoothing,
    get_params2,
    get_random_dataset_directory,
    load_imagenet_dataset,
    make_categories2,
    make_observed_values2,
)
from omegaconf import OmegaConf
from ptflops import get_model_complexity_info
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader


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
    if result_files := glob.glob(str(log_root_dir / "*" / "result.pkl")):
        current_best_accuracy = -float("inf")
        current_best_result_directory = ""
        for log_dir in result_files:
            with open(Path(log_dir).resolve(), "rb") as f:
                result = pickle.load(f)

            if isinstance(result, dict) and "validation_accuracy" in result:
                if result["validation_accuracy"] > current_best_accuracy:
                    current_best_result_directory = log_dir
                    current_best_accuracy = result["validation_accuracy"]

        return Path(current_best_result_directory).resolve()
    raise FileNotFoundError(f"Result files not found: {log_root_dir}")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--retrain-config", type=str, default="retrain_config.yaml")
    args, _ = parser.parse_known_args()
    retrain_config_path = Path(args.retrain_config)

    retrain_config = OmegaConf.load(retrain_config_path)

    retrain_environment = retrain_config.environment
    result_dir = Path(retrain_environment.result_dir).resolve()
    device_ids = None if retrain_environment.device_ids is None else list(map(int, retrain_environment.device_ids))
    num_workers = int(retrain_environment.num_workers)

    retrain_dataset = retrain_config.dataset
    train_data_path = Path(retrain_dataset.train)
    test_data_path = Path(retrain_dataset.test) if retrain_dataset.test else None
    num_classes = int(retrain_dataset.num_classes)

    retrain_hyperparameters = retrain_config.hyperparameters
    batch_size = int(retrain_hyperparameters.batch_size)
    num_epochs = int(retrain_hyperparameters.num_epochs)
    base_lr = float(retrain_hyperparameters.base_lr)
    momentum = float(retrain_hyperparameters.momentum)
    weight_decay = float(retrain_hyperparameters.weight_decay)

    log_dir = Path(result_dir) / f"retrain-{datetime.now().strftime('%d-%m-%Y-%H-%M-%S-%f')}"
    if not log_dir.exists():
        log_dir.mkdir(parents=True)
    logger = create_logger(log_dir, "root.retrain")
    logger.info(str(log_dir))

    try:
        best_result_directory = _get_best_result_directory(result_dir)
    except BaseException as e:
        logger.exception(e)
        raise e

    logger.info(f"Best result of architecture search: {best_result_directory.parent}")

    shutil.copytree(best_result_directory.parent, str(log_dir / "architecture_search_result"))

    with open(best_result_directory, "rb") as f:
        result = pickle.load(f)

    nas_config = result["nas_config"]

    # TODO: Replace "nas/config_*_cifar10.yaml" with "nas/config_*_cifar100.yaml" if retrain_dataset.name == "cifar100".
    nas_parameters = nas_config.nas
    if nas_parameters.search_space == "proxyless" and retrain_dataset.name == "imagenet":
        search_space_config_path = Path("nas/config_proxyless_imagenet.yaml")
    elif nas_parameters.search_space == "proxyless" and retrain_dataset.name == "cifar10":
        search_space_config_path = Path("nas/config_proxyless_cifar10.yaml")
    elif nas_parameters.search_space == "proxyless" and retrain_dataset.name == "cifar100":
        search_space_config_path = Path("nas/config_proxyless_cifar10.yaml")
    elif nas_parameters.search_space == "mnasnet" and retrain_dataset.name == "imagenet":
        search_space_config_path = Path("nas/config_mnasnet_imagenet.yaml")
    elif nas_parameters.search_space == "mnasnet" and retrain_dataset.name == "cifar10":
        search_space_config_path = Path("nas/config_mnasnet_cifar10.yaml")
    elif nas_parameters.search_space == "mnasnet" and retrain_dataset.name == "cifar100":
        search_space_config_path = Path("nas/config_mnasnet_cifar10.yaml")
    else:
        raise ValueError(f"Invalid search space: {nas_parameters.search_space} or dataset: {retrain_dataset.name}")

    trained_theta: list[np.ndarray[Any, np.dtype[float]]] = result["trained_theta"]
    structure_info: MnasNetStructureInfo = result["structure_info"]

    hyperparameters: dict[str, Any] = result["hyperparameters"]
    alpha = hyperparameters["alpha"]
    epsilon = hyperparameters["epsilon"]

    logger.info("Retraining")

    config = create_config_by_yaml(str(search_space_config_path))

    if retrain_dataset.name == "imagenet":
        directory_list, label_map = get_random_dataset_directory(train_data_path, num_classes=num_classes)
        train_transform, test_transform = _data_transforms_imagenet()
        train_dataset = load_imagenet_dataset(
            train_data_path, directory_list, label_map, transform=train_transform, train=True
        )
    elif retrain_dataset.name == "cifar10":
        train_transform, test_transform = _data_transforms_cifar()
        train_dataset = torchvision.datasets.CIFAR10(
            train_data_path, train=True, transform=train_transform, download=True
        )
    elif retrain_dataset.name == "cifar100":
        train_transform, test_transform = _data_transforms_cifar()
        train_dataset = torchvision.datasets.CIFAR100(
            train_data_path, train=True, transform=train_transform, download=True
        )
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)

    if test_data_path is not None and test_data_path.exists() and retrain_dataset.name == "imagenet":
        valid_dataset = load_imagenet_dataset(
            test_data_path, directory_list, label_map, transform=test_transform, train=False
        )
        valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=False, num_workers=num_workers)
    elif test_data_path is not None and test_data_path.exists() and retrain_dataset.name == "cifar10":
        valid_dataset = torchvision.datasets.CIFAR10(
            test_data_path, train=False, transform=test_transform, download=True
        )
        valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=False, num_workers=num_workers)
    elif test_data_path is not None and test_data_path.exists() and retrain_dataset.name == "cifar100":
        valid_dataset = torchvision.datasets.CIFAR100(
            test_data_path, train=False, transform=test_transform, download=True
        )
        valid_dataloader = DataLoader(valid_dataset, batch_size, shuffle=False, num_workers=num_workers)
    else:
        valid_dataloader = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nn_model = MnasNetSearchSpace("MnasNetSearchSpace")
    nn_model.build(config)
    category_dic = nn_model.enumerate_categorical_variables()
    categories = make_categories2(category_dic, config)
    params = get_params2(nn_model, structure_info, config, categories)
    init_delta = 1.0 / sum(categories)

    asng = CategoricalASNG(
        categories, params, alpha=alpha, eps=epsilon, init_delta=init_delta, init_theta=trained_theta
    )

    mle_one_hot = asng.p_model.mle()
    mle = np.argmax(mle_one_hot, axis=1)
    mle_new = make_observed_values2(mle, config, categories)

    structure_info.update_values(mle_new)
    nn_model.select_active_op(structure_info)
    nn_model.fix_arc()

    # nn_model.print_active_op(log_dir=log_dir)
    logger.info(f"The numbers of parameters: {nn_model.get_param_num_list()}")

    p = 0
    for param in nn_model.parameters():
        p += param.numel()

    macs, params_count = get_model_complexity_info(
        nn_model, (3, 224, 224), as_strings=False, print_per_layer_stat=False, verbose=False
    )
    logger.info(f"flops: {macs / 1e6}")

    with open(log_dir / "description.txt", "a") as o:
        o.write("flops(after search)_" + str(epsilon) + ": %fM\n" % (macs / 1e6))

    if device.type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    nn_model = nn.DataParallel(nn_model, device_ids=device_ids)
    nn_model = nn_model.to(device)

    optimizer = SGD(
        nn_model.parameters(),
        lr=base_lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    loss_func = cross_entropy_with_label_smoothing
    nn_model = train(
        nn_model=nn_model,
        asng=asng,
        loss_func=loss_func,
        optimizer=optimizer,
        n_epochs=num_epochs,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        base_lr=base_lr,
        workdir=log_dir,
        parent_logger_name="root.retrain",
    )
    torch.save(nn_model.module.state_dict(), log_dir / "trained_model.pt")


if __name__ == "__main__":
    main()
