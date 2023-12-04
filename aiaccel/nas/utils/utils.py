from __future__ import annotations

import os
import random
from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from _typeshed import FileDescriptorOrPath

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import yaml
from PIL import Image
from torch.utils.data import Dataset

# from torch.utils.tensorboard import SummaryWriter

ParameterType = Union[str, float, int]


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    return device


def get_yaml_value(inputDict: dict[str, dict[str, Any]], key1: str, key2: str) -> Any | None:
    """Gets value from dictionary.

    Args:
        inputDict (dict[str, dict[str, Any]]):
        key1 (str):
        key2 (str):

    Returns:
        Any | None:
    """
    if key1 in inputDict.keys() and key2 in inputDict[key1].keys():
        # print("[" + key1 + "]" + "[" + key2 + "]:" + str(inputDict[key1][key2]))
        return inputDict[key1][key2]
    else:
        print("**Set {}.**".format(key2))
        return None


def load_yaml_file(yaml_file_path: FileDescriptorOrPath) -> Any | int:
    """Loads yaml file.

    Args:
        yaml_file_path (FileDescriptorOrPath):

    Returns:
        Any | int:
    """
    # load yaml file
    if Path(yaml_file_path).is_file() is False:
        raise FileNotFoundError(f"No such file: {yaml_file_path}")

    with open(yaml_file_path, "r") as f:
        param = yaml.load(f, Loader=yaml.SafeLoader)

    return param


def is_valid_yaml_values(
    k_size_range: Any,
    base_layer_num: list[int],
    layer_search_range: Any,
    base_filter_size: list[int],
    filter_size_search_ratio: Any,
    conv_ops: list[str],
    expansion_ratio: Any,
    skip_ops: list[str],
    se_ratio: list[float],
    class_num: Any,
    strides: list[int],
    first_conv_filter_size: Any,
) -> int:
    """Validates parameters.

    Args:
        k_size_range (Any): _description_
        base_layer_num (list[int]): _description_
        layer_search_range (Any): _description_
        base_filter_size (list[int]): _description_
        filter_size_search_ratio (Any): _description_
        conv_ops (list[str]): _description_
        expansion_ratio (Any): _description_
        skip_ops (list[str]): _description_
        se_ratio (list[float]): _description_
        class_num (Any): _description_
        strides (list[int]): _description_
        first_conv_filter_size (Any): _description_

    Returns:
        int: 1 if all parameters are valid, else 0.
    """
    value_list = [
        k_size_range,
        base_layer_num,
        layer_search_range,
        base_filter_size,
        filter_size_search_ratio,
        conv_ops,
        expansion_ratio,
        skip_ops,
        se_ratio,
        class_num,
        strides,
        first_conv_filter_size,
    ]

    if None in value_list:
        return 0

    if not (len(base_layer_num) == len(base_filter_size) == len(strides)):
        print("'base_layer_num', 'base_out_ch' and 'stride' should have same length.")
        return 0

    for conv_op in conv_ops:
        if conv_op not in ["Conv", "SepConv", "MBConv", "Zero"]:
            print("'conv_ops' must be chosen from ['Conv', 'SepConv', 'MBConv'].")
            return 0

    for skip_op in skip_ops:
        if skip_op not in ["no_skip", "skip"]:
            print("'skip' must be chosen from ['no_skip', 'skip'].")
            return 0

    for se in se_ratio:
        if se < 0 or 1 <= se:
            print("Set 'se_ratio' 0 <= se_ratio < 1.")
            return 0

    return 1


def create_config_by_yaml(info_file_path: str) -> tuple[int, dict[str, Any] | None]:
    """Creates config dictionary based on yaml file.

    Args:
        info_file_path (str): yaml file path.

    Returns:
        tuple[int, dict[str, Any] | None]:
    """

    param = load_yaml_file(info_file_path)

    k_size_range = get_yaml_value(param, "search_range", "kernel_size")
    base_layer_num = get_yaml_value(param, "search_range", "base_layer_num")
    layer_search_range = get_yaml_value(param, "search_range", "layer_search_range")
    base_filter_size = get_yaml_value(param, "search_range", "base_filter_size")
    filter_size_search_ratio = get_yaml_value(param, "search_range", "filter_size_search_ratio")
    conv_ops = get_yaml_value(param, "search_range", "conv_ops")
    expansion_ratio = get_yaml_value(param, "search_range", "expansion_ratio")
    skip_ops = get_yaml_value(param, "search_range", "skip_ops")
    se_ratio = get_yaml_value(param, "search_range", "se_ratio")

    class_num = get_yaml_value(param, "settings", "class_num")
    strides = get_yaml_value(param, "settings", "strides")
    first_conv_filter_size = get_yaml_value(param, "settings", "first_conv_filter_size")
    if "proxyless" in info_file_path:
        first_mbconv_filter_size = get_yaml_value(param, "settings", "first_mbconv_filter_size")
    last_ch = get_yaml_value(param, "settings", "last_ch")

    res = is_valid_yaml_values(
        k_size_range,
        base_layer_num,
        layer_search_range,
        base_filter_size,
        filter_size_search_ratio,
        conv_ops,
        expansion_ratio,
        skip_ops,
        se_ratio,
        class_num,
        strides,
        first_conv_filter_size,
    )

    if res == 0:
        raise ValueError("Invalid yaml value.")

    config = {}
    config["class_num"] = class_num
    config["block_num"] = len(base_layer_num)
    config["first_conv_filter_size"] = first_conv_filter_size
    if "proxyless" in info_file_path:
        config["first_mbconv_filter_size"] = first_mbconv_filter_size
    config["last_ch"] = last_ch

    for block_idx in range(len(base_layer_num)):
        key_str_head = "Block{}".format(block_idx)

        layer_stack_num_index_str = key_str_head + "#layer_stack_num_index"
        layer_stack_num_index = [base_layer_num[block_idx] + diff for diff in layer_search_range]
        config[layer_stack_num_index_str] = [i for i in layer_stack_num_index if i > 0]

        key_str_head += "@LayerStack"

        max_in_ch_str = key_str_head + "#max_in_ch"
        if block_idx == 0:
            if "proxyless" not in info_file_path:
                config[max_in_ch_str] = first_conv_filter_size
            else:
                config[max_in_ch_str] = first_mbconv_filter_size
        else:
            config[max_in_ch_str] = int(base_filter_size[block_idx - 1] * np.max(filter_size_search_ratio))

        strides_str = key_str_head + "#stride"
        config[strides_str] = strides[block_idx]

        key_str_head += "@Layer"

        filter_size_index_str = key_str_head + "#filter_size_index"
        config[filter_size_index_str] = [int(base_filter_size[block_idx] * diff) for diff in filter_size_search_ratio]

        if block_idx == len(base_layer_num) - 1:
            config["classifier_in_chs"] = config[filter_size_index_str]

        kernel_size_index_str = key_str_head + "#kernel_size_index"
        config[kernel_size_index_str] = k_size_range

        conv_op_index_str = key_str_head + "#conv_op_index"
        config[conv_op_index_str] = conv_ops

        skip_ops_str = key_str_head + "#skip_op_str"
        config[skip_ops_str] = skip_ops

        se_ratio_str = key_str_head + "#SE_ratio"
        config[se_ratio_str] = se_ratio

        key_str_head_mbconv = key_str_head + "@MBConv"

        ex_ratio_str = key_str_head_mbconv + "#expansion_ratio"
        config[ex_ratio_str] = expansion_ratio

    return config


def make_categories(category_dic, config):
    categories = []
    block_num = int(len(category_dic) / 7)

    for i in range(block_num):
        categories.append(category_dic["Block" + str(i) + "#layer_stack_num_index"])

        if "MBConv" in config["Block" + str(i) + "@LayerStack@Layer#conv_op_index"]:
            value = (
                category_dic["Block" + str(i) + "@LayerStack@Layer#SE_ratio"]
                * category_dic["Block" + str(i) + "@LayerStack@Layer#filter_size_index"]
                * category_dic["Block" + str(i) + "@LayerStack@Layer#kernel_size_index"]
            ) * (
                (category_dic["Block" + str(i) + "@LayerStack@Layer#conv_op_index"] - 1)
                + (category_dic["Block" + str(i) + "@LayerStack@Layer@MBConv#expansion_ratio"])
            )
        else:
            value = (
                category_dic["Block" + str(i) + "@LayerStack@Layer#SE_ratio"]
                * category_dic["Block" + str(i) + "@LayerStack@Layer#conv_op_index"]
                * category_dic["Block" + str(i) + "@LayerStack@Layer#filter_size_index"]
                * category_dic["Block" + str(i) + "@LayerStack@Layer#kernel_size_index"]
            )

        categories.append(value)

        categories.append(category_dic["Block" + str(i) + "@LayerStack@Layer#skip_op_str"])

    return np.array(categories)


def make_categories2(category_dic, config):
    categories = []
    block_num = int(len(category_dic) / 7)

    for i in range(block_num):
        value = (
            category_dic["Block" + str(i) + "@LayerStack@Layer#SE_ratio"]
            * category_dic["Block" + str(i) + "@LayerStack@Layer#filter_size_index"]
            * category_dic["Block" + str(i) + "@LayerStack@Layer#kernel_size_index"]
        ) * ((category_dic["Block" + str(i) + "@LayerStack@Layer@MBConv#expansion_ratio"]))
        if i > 0:
            ch0 = config["Block" + str(i - 1) + "@LayerStack@Layer#filter_size_index"][0]
            ch1 = config["Block" + str(i) + "@LayerStack@Layer#filter_size_index"][0]
            st = config["Block" + str(i) + "@LayerStack#stride"]
            if ch0 == ch1 and st == 1:
                value += 1
        categories.append(value)

    return np.array(categories)


def make_observed_values1(observed_values, config, categories):
    observed_values_new = []
    block_num = int(len(observed_values) / 3)
    for i in range(block_num):
        # レイヤー数追加
        observed_values_new.append(observed_values[3 * i])

        index = observed_values[3 * i + 1]
        index2 = categories[3 * i + 1]

        se_ratio_index = (
            np.digitize(
                index,
                bins=[
                    (index2 // len(config["Block" + str(i) + "@LayerStack@Layer#SE_ratio"])) * j
                    for j in range(len(config["Block" + str(i) + "@LayerStack@Layer#SE_ratio"]))
                ],
            )
            - 1
        )
        index2 = index2 // len(config["Block" + str(i) + "@LayerStack@Layer#SE_ratio"])
        index -= se_ratio_index * index2

        filter_size_index = (
            np.digitize(
                index,
                bins=[
                    (index2 // len(config["Block" + str(i) + "@LayerStack@Layer#filter_size_index"])) * j
                    for j in range(len(config["Block" + str(i) + "@LayerStack@Layer#filter_size_index"]))
                ],
            )
            - 1
        )
        index2 = index2 // len(config["Block" + str(i) + "@LayerStack@Layer#filter_size_index"])
        index -= filter_size_index * index2

        kernel_size_index = (
            np.digitize(
                index,
                bins=[
                    (index2 // len(config["Block" + str(i) + "@LayerStack@Layer#kernel_size_index"])) * j
                    for j in range(len(config["Block" + str(i) + "@LayerStack@Layer#kernel_size_index"]))
                ],
            )
            - 1
        )
        index2 = index2 // len(config["Block" + str(i) + "@LayerStack@Layer#kernel_size_index"])
        index -= kernel_size_index * index2

        # MBConvが含まれるとき
        if "MBConv" in config["Block" + str(i) + "@LayerStack@Layer#conv_op_index"]:
            n = (
                len(config["Block" + str(i) + "@LayerStack@Layer#conv_op_index"])
                - 1
                + len(config["Block" + str(i) + "@LayerStack@Layer@MBConv#expansion_ratio"])
            )
            conv_op_index = np.digitize(index, bins=[(index2 // n) * j for j in range(n)]) - 1

            # MBConv以外が選ばれたとき
            if conv_op_index < len(config["Block" + str(i) + "@LayerStack@Layer#conv_op_index"]) - 1:
                expansion_ratio_index = 0

            else:
                expansion_ratio_index = (
                    conv_op_index - len(config["Block" + str(i) + "@LayerStack@Layer#conv_op_index"]) + 1
                )
                conv_op_index = len(config["Block" + str(i) + "@LayerStack@Layer#conv_op_index"]) - 1

        # MBConvが含まれないとき
        else:
            conv_op_index = (
                np.digitize(
                    index,
                    bins=[
                        (index2 // len(config["Block" + str(i) + "@LayerStack@Layer#conv_op_index"])) * j
                        for j in range(len(config["Block" + str(i) + "@LayerStack@Layer#conv_op_index"]))
                    ],
                )
                - 1
            )

        observed_values_new.append(filter_size_index)
        observed_values_new.append(kernel_size_index)
        observed_values_new.append(conv_op_index)
        # skip有無追加
        observed_values_new.append(observed_values[3 * i + 2])
        observed_values_new.append(se_ratio_index)

        observed_values_new.append(expansion_ratio_index)

    return np.array(observed_values_new)


def make_observed_values2(observed_values, config, categories):
    observed_values_new = []
    block_num = int(len(observed_values))
    prev_filter_size = -1
    for i in range(block_num):
        # レイヤー数追加
        observed_values_new.append(0)

        index = observed_values[i]
        index2 = categories[i]

        ch = config["Block" + str(i) + "@LayerStack@Layer#filter_size_index"][0]
        st = config["Block" + str(i) + "@LayerStack#stride"]

        # "Zero"が適用される可能性がある場合
        if prev_filter_size == ch and st == 1:
            if index == 0:
                observed_values_new.append(0)
                observed_values_new.append(0)
                observed_values_new.append(0)
                # skip有無追加
                observed_values_new.append(0)
                observed_values_new.append(0)
                observed_values_new.append(0)
                continue

            else:
                index -= 1

        kernel_size_index = (
            np.digitize(
                index,
                bins=[
                    (index2 // len(config["Block" + str(i) + "@LayerStack@Layer#kernel_size_index"])) * j
                    for j in range(len(config["Block" + str(i) + "@LayerStack@Layer#kernel_size_index"]))
                ],
            )
            - 1
        )
        index2 = index2 // len(config["Block" + str(i) + "@LayerStack@Layer#kernel_size_index"])
        index -= kernel_size_index * index2

        n = len(config["Block" + str(i) + "@LayerStack@Layer@MBConv#expansion_ratio"])
        expansion_ratio_index = np.digitize(index, bins=[(index2 // n) * j for j in range(n)]) - 1

        conv_op_index = len(config["Block" + str(i) + "@LayerStack@Layer#conv_op_index"]) - 1

        observed_values_new.append(0)
        observed_values_new.append(kernel_size_index)
        observed_values_new.append(conv_op_index)
        # skip有無追加
        observed_values_new.append(0)
        observed_values_new.append(0)
        observed_values_new.append(expansion_ratio_index)

        prev_filter_size = ch

    return np.array(observed_values_new)


def make_observed_values3(observed_values, config, categories):
    observed_values_new = []
    block_num = int(len(observed_values))
    prev_filter_size = -1

    for i in range(block_num):
        # レイヤー数追加
        observed_values_new.append(0)

        index = observed_values[i]
        index2 = categories[i]

        ch = config["Block" + str(i) + "@LayerStack@Layer#filter_size_index"][0]
        st = config["Block" + str(i) + "@LayerStack#stride"]

        # "Zero"が適用される可能性がある場合
        if prev_filter_size == ch and st == 1:
            if index == 0:
                observed_values_new.append(0)
                observed_values_new.append(0)
                observed_values_new.append(0)
                # skip有無追加
                observed_values_new.append(0)
                observed_values_new.append(0)
                observed_values_new.append(0)
                continue

            else:
                index -= 1

        filter_size_index = (
            np.digitize(
                index,
                bins=[
                    (index2 // len(config["Block" + str(i) + "@LayerStack@Layer#filter_size_index"])) * j
                    for j in range(len(config["Block" + str(i) + "@LayerStack@Layer#filter_size_index"]))
                ],
            )
            - 1
        )
        index2 = index2 // len(config["Block" + str(i) + "@LayerStack@Layer#filter_size_index"])
        index -= filter_size_index * index2

        kernel_size_index = (
            np.digitize(
                index,
                bins=[
                    (index2 // len(config["Block" + str(i) + "@LayerStack@Layer#kernel_size_index"])) * j
                    for j in range(len(config["Block" + str(i) + "@LayerStack@Layer#kernel_size_index"]))
                ],
            )
            - 1
        )
        index2 = index2 // len(config["Block" + str(i) + "@LayerStack@Layer#kernel_size_index"])
        index -= kernel_size_index * index2

        n = len(config["Block" + str(i) + "@LayerStack@Layer@MBConv#expansion_ratio"])
        expansion_ratio_index = np.digitize(index, bins=[(index2 // n) * j for j in range(n)]) - 1

        conv_op_index = len(config["Block" + str(i) + "@LayerStack@Layer#conv_op_index"]) - 1

        observed_values_new.append(filter_size_index)
        observed_values_new.append(kernel_size_index)
        observed_values_new.append(conv_op_index)
        # skip有無追加
        observed_values_new.append(0)
        observed_values_new.append(0)
        observed_values_new.append(expansion_ratio_index)

        prev_filter_size = ch

    return np.array(observed_values_new)


def get_params(nn_model, structure_info, config, categories):
    C = np.zeros([len(categories), max(categories)])
    for i in range(len(categories) // 3):
        # レイヤー数に対応するパラメータ数を設定
        C[3 * i, : categories[3 * i]] = config["Block" + str(i) + "#layer_stack_num_index"]

        # オペレーションに対応するパラメータ数を設定
        for j in range(C.shape[1]):
            observed_values = np.zeros([C.shape[0]], dtype=np.int64)

            observed_values[3 * i + 1] = j
            observed_values2 = make_observed_values1(observed_values, config, categories)

            if i == 0:
                structure_info.update_values(observed_values2)

                nn_model.select_active_op(structure_info)
                params = nn_model.get_param_num_list()
                C[3 * i + 1, j] = params[1][0]
            else:
                v = 0
                for k in range(len(config["Block" + str(i - 1) + "@LayerStack@Layer#filter_size_index"])):
                    observed_values2[7 * i - 6] = k
                    structure_info.update_values(observed_values2)
                    nn_model.select_active_op(structure_info)
                    params = nn_model.get_param_num_list()
                    v += params[i + 1][0]
                C[3 * i + 1, j] = v / len(config["Block" + str(i - 1) + "@LayerStack@Layer#filter_size_index"])

    return C


def get_params2(nn_model, structure_info, config, categories):
    C = np.zeros([len(categories), max(categories)])
    for i in range(len(categories)):
        # オペレーションに対応するパラメータ数を設定
        for j in range(categories[i]):
            observed_values = np.zeros([C.shape[0]], dtype=np.int64)

            observed_values[i] = j
            observed_values2 = make_observed_values2(observed_values, config, categories)

            structure_info.update_values(observed_values2)
            nn_model.select_active_op(structure_info)
            params = nn_model.get_param_num_list()
            C[i, j] = params[i + 1][0]

    return C


def get_params3(nn_model, structure_info, config, categories):
    C = np.zeros([len(categories), max(categories)])
    for i in range(len(categories)):
        # オペレーションに対応するパラメータ数を設定
        for j in range(categories[i]):
            observed_values = np.zeros([C.shape[0]], dtype=np.int64)

            observed_values[i] = j
            observed_values2 = make_observed_values3(observed_values, config, categories)

            if i == 0:
                structure_info.update_values(observed_values2)

                nn_model.select_active_op(structure_info)
                params = nn_model.get_param_num_list()
                C[i, j] = params[1][0]
            else:
                v = 0
                for k in range(len(config["Block" + str(i - 1) + "@LayerStack@Layer#filter_size_index"])):
                    observed_values2[7 * i - 6] = k
                    structure_info.update_values(observed_values2)
                    nn_model.select_active_op(structure_info)
                    params = nn_model.get_param_num_list()
                    v += params[i + 1][0]
                C[i, j] = v / len(config["Block" + str(i - 1) + "@LayerStack@Layer#filter_size_index"])

    return C


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.1):
    logsoftmax = nn.LogSoftmax(dim=1)
    n_classes = pred.size(1)
    # convert to one-hot
    target = torch.unsqueeze(target, 1)
    soft_target = torch.zeros_like(pred)
    soft_target.scatter_(1, target, 1)
    # label smoothing
    soft_target = soft_target * (1 - label_smoothing) + label_smoothing / n_classes
    return torch.mean(torch.sum(-soft_target * logsoftmax(pred), 1))


class ImageNetCustomDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        with open(self.df["id"].iloc[idx], "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, self.df["label"].iloc[idx]


def get_random_dataset_directory(
    path: FileDescriptorOrPath, num_classes: int | None = None
) -> tuple[list[str], dict[str, int]]:
    """Reduce the number of classes in the dataset and returns a list of
    directories and a dict to map directory to the corresponding labels.

    Args:
        path (FileDescriptorOrPath): A path to parent directory of dataset.
        num_classes (int | None, optional): The number of classes to be
            classified. If None, all of classes are used. Defaults to None.

    Returns:
        tuple[list[str], dict[str, int]]: A list of directory names and a dict to
            map directory names to the corresponding label integers.
    """
    directory_list = os.listdir(path)
    if num_classes is None or num_classes > len(directory_list):
        num_classes = len(directory_list)
    directory_list = random.sample(directory_list, num_classes)
    label_map = {f: i for i, f in enumerate(directory_list)}
    return directory_list, label_map


def load_imagenet_dataset(
    path: FileDescriptorOrPath,
    directory_list: list[str],
    label_map: dict[str, int],
    transform: Any | None = None,
    train: bool = True,
) -> ImageNetCustomDataset:
    """Loads imagenet dataset.

    Args:
        path (FileDescriptorOrPath): A path to parent directory of dataset.
        directory_list (list[str]): A list of directory names.
        label_map (dict[str, int]): A dict to map directory names on the
            corresponding labels.
        transform (Any | None, optional): Data transform. Defaults to None.
        train (bool, optional): Whether dataset is for train or not. Defaults
            to True.

    Returns:
        ImageNetCustomDataset: A Dataset of imagenet.
    """
    file_list = []
    label_list = []
    for file in directory_list:
        data = os.listdir(os.path.join(path, file))
        file_list += [os.path.join(path, file, f) for f in data]
        label_list += [label_map[file]] * len(data)
    dataframe = pd.DataFrame({"id": file_list, "label": label_list})
    if train:
        dataframe = dataframe.sample(frac=1)
    dataset = ImageNetCustomDataset(dataframe, transform=transform)
    return dataset


def get_random_indices(num_all_data: int, num_subset_data: int) -> tuple[list[int], list[int]]:
    """Gets lists of indices specifying data in each subset.

    The size of a subset is specified by num_subset_data and that of the
    other is num_all_data - num_subset_data.

    Args:
        num_all_data (int): The number of all data.
        num_subset_data (int): The number of data in a subset.

    Returns:
        tuple[list[int], list[int]]: Lists of indices. The size of former is
            num_subset_data.
    """
    data_indices = list(range(num_all_data))
    subset_indices = random.sample(data_indices, num_subset_data)
    remained_subset_indices = list(set(data_indices) - set(subset_indices))
    return subset_indices, remained_subset_indices


def imagenet_dataset(train_path, valid_path, train_transforms=None, test_transforms=None, is_search=True):
    directory_list = os.listdir(train_path)
    if is_search:
        directory_list = random.sample(directory_list, 200)
    label_map = {f: i for i, f in enumerate(directory_list)}

    train_file_list = []
    train_label_list = []
    for i, file in enumerate(directory_list):
        data = os.listdir(os.path.join(train_path, file))
        train_file_list += [os.path.join(train_path, file, f) for f in data]
        train_label_list += [label_map[file] for k in range(len(data))]
    train_data = pd.DataFrame({"id": train_file_list, "label": train_label_list})
    train_data = train_data.sample(frac=1)
    train_dataset = ImageNetCustomDataset(train_data, transform=train_transforms)

    test_file_list = []
    test_label_list = []
    for i, file in enumerate(directory_list):
        data = os.listdir(os.path.join(valid_path, file))
        test_file_list += [os.path.join(valid_path, file, f) for f in data]
        test_label_list += [label_map[file] for k in range(len(data))]
    test_data = pd.DataFrame({"id": test_file_list, "label": test_label_list})

    test_dataset = ImageNetCustomDataset(test_data, transform=test_transforms)

    return train_dataset, test_dataset


def _data_transforms_imagenet():
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, (0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    test_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    return train_transforms, test_transforms


def _data_transforms_cifar():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transforms = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )
    return train_transforms, test_transforms


def drop_path(X, drop_prob):
    assert 0.0 <= drop_prob and drop_prob < 1.0
    if drop_prob > 0.0:
        keep_prob = 1.0 - drop_prob
        mask = torch.FloatTensor(X.size(0), 1, 1, 1).bernoulli_(keep_prob)
        if X.is_cuda:
            mask = mask.cuda(X.get_device())
        X = X * mask / keep_prob
    return X


# def create_tensorboard_log(
#     model: nn.DataParallel,
#     dataloader: torch.utils.data.DataLoader,
#     device: torch.device,
#     logdir: FileDescriptorOrPath,
# ) -> None:
#     data = dataloader.__iter__()
#     image, _ = next(data)
#     image = image.to(device)
#     with SummaryWriter(log_dir=str(logdir / "tb_logs/model")) as w:
#         w.add_graph(model, image)


def get_search_space_config(search_space_name: str = "proxyless", dataset_name: str = "imagenet") -> Path:
    # TODO: Replace "nas/config_*_cifar10.yaml" with "nas/config_*_cifar100" if nas_dataset.name == "cifar100".
    if search_space_name == "proxyless" and dataset_name == "imagenet":
        return Path("./config_proxyless_imagenet.yaml")
    elif search_space_name == "proxyless" and dataset_name == "cifar10":
        return Path("./config_proxyless_cifar10.yaml")
    elif search_space_name == "proxyless" and dataset_name == "cifar100":
        return Path("./config_proxyless_cifar10.yaml")
    elif search_space_name == "mnasnet" and dataset_name == "imagenet":
        return Path("./config_mnasnet_imagenet.yaml")
    elif search_space_name == "mnasnet" and dataset_name == "cifar10":
        return Path("./config_mnasnet_cifar10.yaml")
    elif search_space_name == "mnasnet" and dataset_name == "cifar100":
        return Path("./config_mnasnet_cifar10.yaml")
    else:
        raise ValueError(f"Invalid search space: {search_space_name} or dataset: {dataset_name}")
