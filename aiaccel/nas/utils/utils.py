from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Callable, Union

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch import Tensor, device, nn
from torch.utils.data import Dataset
from torchvision import transforms

ParameterType = Union[str, float, int]


def get_device() -> device:
    """
    Get the available device for torch.

    Returns:
        Device: The available device. It could be CUDA, MPS, or CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def get_yaml_value(input_dict: dict[str, dict[str, Any]], key1: str, key2: str) -> Any | None:
    """
    Gets a nested value from a dictionary.

    Args:
        input_dict (Dict[str, Dict[str, Any]]): The input dictionary.
        key1 (str): The first level key.
        key2 (str): The second level key.

    Returns:
        Optional[Any]: The value at the specified keys if they exist, None otherwise.
    """
    if key1 in input_dict and key2 in input_dict[key1]:
        return input_dict[key1][key2]

    print(f"**Set {key2}.**")

    return None


def load_yaml_file(yaml_file_path: str | Path) -> Any:
    """
    Loads yaml file.

    Args:
        yaml_file_path (Union[str, Path]): The path to the yaml file.

    Returns:
        Any: The content of the yaml file.

    Raises:
        FileNotFoundError: If the yaml file does not exist.
    """
    yaml_path = Path(yaml_file_path)

    if not yaml_path.is_file():
        raise FileNotFoundError(yaml_file_path)

    # with Path.open(yaml_file_path) as f:
    with Path.open(yaml_path) as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


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
    """
    Validates parameters.

    Args:
        k_size_range (Any): Kernel size range.
        base_layer_num (List[int]): Base layer numbers.
        layer_search_range (Any): Layer search range.
        base_filter_size (List[int]): Base filter sizes.
        filter_size_search_ratio (Any): Filter size search ratio.
        conv_ops (List[str]): Convolution operations.
        expansion_ratio (Any): Expansion ratio.
        skip_ops (List[str]): Skip operations.
        se_ratio (List[float]): Squeeze and Excitation ratio.
        class_num (Any): Number of classes.
        strides (List[int]): Strides.
        first_conv_filter_size (Any): First convolution filter size.

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
        if se < 0 or se >= 1:
            print("Set 'se_ratio' 0 <= se_ratio < 1.")
            return 0

    return 1


def create_config_by_yaml(info_file_path: str) -> tuple[int, dict[str, int | list[int] | str | list[str]]]:
    """
    Creates config dictionary based on yaml file.

    Args:
        info_file_path (str): yaml file path.

    Returns:
        tuple[int, dict[str, int | list[int] | str | list[str]]]: Tuple containing an integer and a dictionary.
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
    first_mbconv_filter_size = (
        get_yaml_value(param, "settings", "first_mbconv_filter_size") if "proxyless" in info_file_path else None
    )
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
        raise ValueError(res)

    config = {
        "class_num": class_num,
        "block_num": len(base_layer_num),
        "first_conv_filter_size": first_conv_filter_size,
        "first_mbconv_filter_size": first_mbconv_filter_size if "proxyless" in info_file_path else None,
        "last_ch": last_ch,
    }

    for block_idx in range(len(base_layer_num)):
        key_str_head = f"Block{block_idx}"

        layer_stack_num_index_str = key_str_head + "#layer_stack_num_index"
        layer_stack_num_index = [base_layer_num[block_idx] + diff for diff in layer_search_range]
        config[layer_stack_num_index_str] = [i for i in layer_stack_num_index if i > 0]

        key_str_head += "@LayerStack"

        max_in_ch_str = key_str_head + "#max_in_ch"
        if block_idx == 0:
            config[max_in_ch_str] = (
                first_mbconv_filter_size if "proxyless" in info_file_path else first_conv_filter_size
            )
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


def make_categories(category_dic: dict[str, Any], config: dict[str, Any]) -> np.ndarray:
    """
    Makes categories based on the category dictionary and configuration.

    Args:
        category_dic (dict[str, Any]): The category dictionary.
        config (dict[str, Any]): The configuration dictionary.

    Returns:
        np.ndarray: The categories as a numpy array.
    """
    categories: list[Any] = []
    block_num = len(category_dic) // 7

    for i in range(block_num):
        block_key = f"Block{i}"
        categories.append(category_dic[f"{block_key}#layer_stack_num_index"])

        if "MBConv" in config[f"{block_key}@LayerStack@Layer#conv_op_index"]:
            value = (
                category_dic[f"{block_key}@LayerStack@Layer#SE_ratio"]
                * category_dic[f"{block_key}@LayerStack@Layer#filter_size_index"]
                * category_dic[f"{block_key}@LayerStack@Layer#kernel_size_index"]
            ) * (
                (category_dic[f"{block_key}@LayerStack@Layer#conv_op_index"] - 1)
                + (category_dic[f"{block_key}@LayerStack@Layer@MBConv#expansion_ratio"])
            )
        else:
            value = (
                category_dic[f"{block_key}@LayerStack@Layer#SE_ratio"]
                * category_dic[f"{block_key}@LayerStack@Layer#conv_op_index"]
                * category_dic[f"{block_key}@LayerStack@Layer#filter_size_index"]
                * category_dic[f"{block_key}@LayerStack@Layer#kernel_size_index"]
            )

        categories.append(value)
        categories.append(category_dic[f"{block_key}@LayerStack@Layer#skip_op_str"])

    return np.array(categories)


def make_categories2(category_dic: dict[str, Any], config: dict[str, Any]) -> np.ndarray:
    """
    Create an array of categories based on the given category dictionary and configuration.

    Args:
        category_dic (dict[str, Any]): A dictionary containing category information.
        config (dict[str, Any]): A configuration dictionary with specific settings.

    Returns:
        np.ndarray: An array of calculated category values.
    """
    block_num = len(category_dic) // 7
    categories = []

    for i in range(block_num):
        block_key = f"Block{i}@LayerStack@Layer"
        value = (
            category_dic[block_key + "#SE_ratio"]
            * category_dic[block_key + "#filter_size_index"]
            * category_dic[block_key + "#kernel_size_index"]
            * category_dic[block_key + "@MBConv#expansion_ratio"]
        )

        if i > 0:
            prev_block_key = f"Block{i - 1}@LayerStack@Layer"
            ch0 = config[prev_block_key + "#filter_size_index"][0]
            ch1 = config[block_key + "#filter_size_index"][0]
            st = config[f"Block{i}@LayerStack#stride"]

            if ch0 == ch1 and st == 1:
                value += 1

        categories.append(value)

    return np.array(categories)


def make_observed_values1(observed_values: np.ndarray, config: dict, categories: np.ndarray) -> np.ndarray:
    """
    Makes observed values based on the observed values, configuration, and categories.

    Args:
        observed_values (np.ndarray): The observed values.
        config (dict): The configuration dictionary.
        categories (np.ndarray): The categories.

    Returns:
        np.ndarray: The new observed values.
    """
    observed_values_new = []
    block_num = len(observed_values) // 3

    for i in range(block_num):
        observed_values_new.append(observed_values[3 * i])

        index = observed_values[3 * i + 1]
        index2 = categories[3 * i + 1]

        se_ratio_index = (
            np.digitize(
                index,
                bins=[
                    (index2 // len(config[f"Block{i}@LayerStack@Layer#SE_ratio"])) * j
                    for j in range(len(config[f"Block{i}@LayerStack@Layer#SE_ratio"]))
                ],
            )
            - 1
        )
        index2 = index2 // len(config[f"Block{i}@LayerStack@Layer#SE_ratio"])
        index -= se_ratio_index * index2

        filter_size_index = (
            np.digitize(
                index,
                bins=[
                    (index2 // len(config[f"Block{i}@LayerStack@Layer#filter_size_index"])) * j
                    for j in range(len(config[f"Block{i}@LayerStack@Layer#filter_size_index"]))
                ],
            )
            - 1
        )
        index2 = index2 // len(config[f"Block{i}@LayerStack@Layer#filter_size_index"])
        index -= filter_size_index * index2

        kernel_size_index = (
            np.digitize(
                index,
                bins=[
                    (index2 // len(config[f"Block{i}@LayerStack@Layer#kernel_size_index"])) * j
                    for j in range(len(config[f"Block{i}@LayerStack@Layer#kernel_size_index"]))
                ],
            )
            - 1
        )
        index2 = index2 // len(config[f"Block{i}@LayerStack@Layer#kernel_size_index"])
        index -= kernel_size_index * index2

        if "MBConv" in config[f"Block{i}@LayerStack@Layer#conv_op_index"]:
            n = (
                len(config[f"Block{i}@LayerStack@Layer#conv_op_index"])
                - 1
                + len(config[f"Block{i}@LayerStack@Layer@MBConv#expansion_ratio"])
            )
            conv_op_index = np.digitize(index, bins=[(index2 // n) * j for j in range(n)]) - 1
        else:
            conv_op_index = (
                np.digitize(
                    index,
                    bins=[
                        (index2 // len(config[f"Block{i}@LayerStack@Layer#conv_op_index"])) * j
                        for j in range(len(config[f"Block{i}@LayerStack@Layer#conv_op_index"]))
                    ],
                )
                - 1
            )

        if conv_op_index < len(config[f"Block{i}@LayerStack@Layer#conv_op_index"]) - 1:
            expansion_ratio_index = 0
        else:
            expansion_ratio_index = conv_op_index - len(config[f"Block{i}@LayerStack@Layer#conv_op_index"]) + 1
            conv_op_index = len(config[f"Block{i}@LayerStack@Layer#conv_op_index"]) - 1

        observed_values_new.extend(
            [
                filter_size_index,
                kernel_size_index,
                conv_op_index,
                observed_values[3 * i + 2],
                se_ratio_index,
                expansion_ratio_index,
            ],
        )

    return np.array(observed_values_new)


def make_observed_values2(observed_values: np.ndarray, config: dict, categories: np.ndarray) -> np.ndarray:
    """
    Makes observed values based on the observed values, configuration, and categories.

    Args:
        observed_values (np.ndarray): The observed values.
        config (dict): The configuration dictionary.
        categories (np.ndarray): The categories.

    Returns:
        np.ndarray: The new observed values.
    """
    observed_values_new = []
    block_num = len(observed_values)
    prev_filter_size = -1

    for i in range(block_num):
        observed_values_new.append(0)

        index = observed_values[i]
        index2 = categories[i]

        ch = config[f"Block{i}@LayerStack@Layer#filter_size_index"][0]
        st = config[f"Block{i}@LayerStack#stride"]

        if prev_filter_size == ch and st == 1:
            if index == 0:
                observed_values_new.extend([0, 0, 0, 0, 0, 0])
                continue

            index -= 1

        kernel_size_index = (
            np.digitize(
                index,
                bins=[
                    (index2 // len(config[f"Block{i}@LayerStack@Layer#kernel_size_index"])) * j
                    for j in range(len(config[f"Block{i}@LayerStack@Layer#kernel_size_index"]))
                ],
            )
            - 1
        )
        index2 = index2 // len(config[f"Block{i}@LayerStack@Layer#kernel_size_index"])
        index -= kernel_size_index * index2

        n = len(config[f"Block{i}@LayerStack@Layer@MBConv#expansion_ratio"])
        expansion_ratio_index = np.digitize(index, bins=[(index2 // n) * j for j in range(n)]) - 1

        conv_op_index = len(config[f"Block{i}@LayerStack@Layer#conv_op_index"]) - 1

        observed_values_new.extend([0, kernel_size_index, conv_op_index, 0, 0, expansion_ratio_index])

        prev_filter_size = ch

    return np.array(observed_values_new)


def make_observed_values3(observed_values: np.ndarray, config: dict, categories: np.ndarray) -> np.ndarray:
    """
    Makes observed values based on the observed values, configuration, and categories.

    Args:
        observed_values (np.ndarray): The observed values.
        config (dict): The configuration dictionary.
        categories (np.ndarray): The categories.

    Returns:
        np.ndarray: The new observed values.
    """
    observed_values_new = []
    block_num = len(observed_values)
    prev_filter_size = -1

    for i in range(block_num):
        observed_values_new.append(0)

        index = observed_values[i]
        index2 = categories[i]

        ch = config[f"Block{i}@LayerStack@Layer#filter_size_index"][0]
        st = config[f"Block{i}@LayerStack#stride"]

        if prev_filter_size == ch and st == 1:
            if index == 0:
                observed_values_new.extend([0, 0, 0, 0, 0, 0])
                continue

            index -= 1

        filter_size_index = (
            np.digitize(
                index,
                bins=[
                    (index2 // len(config[f"Block{i}@LayerStack@Layer#filter_size_index"])) * j
                    for j in range(len(config[f"Block{i}@LayerStack@Layer#filter_size_index"]))
                ],
            )
            - 1
        )
        index2 = index2 // len(config[f"Block{i}@LayerStack@Layer#filter_size_index"])
        index -= filter_size_index * index2

        kernel_size_index = (
            np.digitize(
                index,
                bins=[
                    (index2 // len(config[f"Block{i}@LayerStack@Layer#kernel_size_index"])) * j
                    for j in range(len(config[f"Block{i}@LayerStack@Layer#kernel_size_index"]))
                ],
            )
            - 1
        )
        index2 = index2 // len(config[f"Block{i}@LayerStack@Layer#kernel_size_index"])
        index -= kernel_size_index * index2

        n = len(config[f"Block{i}@LayerStack@Layer@MBConv#expansion_ratio"])
        expansion_ratio_index = np.digitize(index, bins=[(index2 // n) * j for j in range(n)]) - 1

        conv_op_index = len(config[f"Block{i}@LayerStack@Layer#conv_op_index"]) - 1

        observed_values_new.extend([filter_size_index, kernel_size_index, conv_op_index, 0, 0, expansion_ratio_index])

        prev_filter_size = ch

    return np.array(observed_values_new)


def get_params(nn_model: object, structure_info: object, config: dict, categories: np.ndarray) -> np.ndarray:
    """
    Get parameters from the neural network model based on the structure information, configuration, and categories.

    Args:
        nn_model (object): The neural network model.
        structure_info (object): The structure information.
        config (dict): The configuration dictionary.
        categories (np.ndarray): The categories.

    Returns:
        np.ndarray: The parameters.
    """
    c = np.zeros([len(categories), max(categories)])

    for i in range(len(categories) // 3):
        # Set the number of parameters corresponding to the number of layers
        c[3 * i, : categories[3 * i]] = config["Block" + str(i) + "#layer_stack_num_index"]

        # Set the number of parameters corresponding to the operation
        for j in range(c.shape[1]):
            observed_values = np.zeros([c.shape[0]], dtype=np.int64)

            observed_values[3 * i + 1] = j
            observed_values2 = make_observed_values1(observed_values, config, categories)

            if i == 0:
                structure_info.update_values(observed_values2)

                nn_model.select_active_op(structure_info)
                params = nn_model.get_param_num_list()
                c[3 * i + 1, j] = params[1][0]
            else:
                v = 0
                for k in range(len(config["Block" + str(i - 1) + "@LayerStack@Layer#filter_size_index"])):
                    observed_values2[7 * i - 6] = k
                    structure_info.update_values(observed_values2)
                    nn_model.select_active_op(structure_info)
                    params = nn_model.get_param_num_list()
                    v += params[i + 1][0]
                c[3 * i + 1, j] = v / len(config["Block" + str(i - 1) + "@LayerStack@Layer#filter_size_index"])

    return c


def get_params2(nn_model: object, structure_info: object, config: dict, categories: np.ndarray) -> np.ndarray:
    """
    Get parameters from the neural network model based on the structure information, configuration, and categories.

    Args:
        nn_model (object): The neural network model.
        structure_info (object): The structure information.
        config (dict): The configuration dictionary.
        categories (np.ndarray): The categories.

    Returns:
        np.ndarray: The parameters.
    """
    c = np.zeros([len(categories), max(categories)])

    for i in range(len(categories)):
        # Set the number of parameters corresponding to the operation
        for j in range(categories[i]):
            observed_values = np.zeros([c.shape[0]], dtype=np.int64)

            observed_values[i] = j
            observed_values2 = make_observed_values2(observed_values, config, categories)

            structure_info.update_values(observed_values2)
            nn_model.select_active_op(structure_info)
            params = nn_model.get_param_num_list()
            c[i, j] = params[i + 1][0]

    return c


def get_params3(nn_model: object, structure_info: object, config: dict, categories: np.ndarray) -> np.ndarray:
    """
    Get parameters from the neural network model based on the structure information, configuration, and categories.

    Args:
        nn_model (object): The neural network model.
        structure_info (object): The structure information.
        config (dict): The configuration dictionary.
        categories (np.ndarray): The categories.

    Returns:
        np.ndarray: The parameters.
    """
    c = np.zeros([len(categories), max(categories)])

    for i in range(len(categories)):
        # Set the number of parameters corresponding to the operation
        for j in range(categories[i]):
            observed_values = np.zeros([c.shape[0]], dtype=np.int64)

            observed_values[i] = j
            observed_values2 = make_observed_values3(observed_values, config, categories)

            if i == 0:
                structure_info.update_values(observed_values2)

                nn_model.select_active_op(structure_info)
                params = nn_model.get_param_num_list()
                c[i, j] = params[1][0]
            else:
                v = 0
                for k in range(len(config["Block" + str(i - 1) + "@LayerStack@Layer#filter_size_index"])):
                    observed_values2[7 * i - 6] = k
                    structure_info.update_values(observed_values2)
                    nn_model.select_active_op(structure_info)
                    params = nn_model.get_param_num_list()
                    v += params[i + 1][0]
                c[i, j] = v / len(config["Block" + str(i - 1) + "@LayerStack@Layer#filter_size_index"])

    return c


def accuracy(output: Tensor, target: Tensor, topk: tuple[int, ...] = (1,)) -> list[Tensor]:
    """
    Computes the accuracy over the k top predictions for the specified values of k.

    Args:
        output (Tensor): The output tensor.
        target (Tensor): The target tensor.
        topk (tuple[int, ...]): The specified values of k.

    Returns:
        list[Tensor]: The accuracies for each value of k.
    """
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


def cross_entropy_with_label_smoothing(pred: Tensor, target: Tensor, label_smoothing: float = 0.1) -> Tensor:
    """
    Computes the cross entropy loss with label smoothing.

    Args:
        pred (Tensor): The predictions.
        target (Tensor): The target values.
        label_smoothing (float, optional): The label smoothing factor. Defaults to 0.1.

    Returns:
        Tensor: The computed loss.
    """
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
    """
    A custom Dataset class for ImageNet.

    Args:
        df (pd.DataFrame): The dataframe containing the image ids and labels.
        transform (Callable, optional): The optional transform to be applied on the images.
    """

    def __init__(self, df: pd.DataFrame, transform: Callable | None = None):
        self.df = df
        self.transform = transform

    def __len__(self) -> int:
        """Returns the length of the dataframe."""
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns the image and label at the specified index.

        Args:
            idx (int): The index.

        Returns:
            tuple: The image and label.
        """
        with Path.open(self.df["id"].iloc[idx], "rb") as f:
            img = Image.open(f)
            img = img.convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, self.df["label"].iloc[idx]


def get_random_dataset_directory(path: str | Path, num_classes: int | None = None) -> tuple[list[str], dict[str, int]]:
    """
    Reduce the number of classes in the dataset and returns a list of
    directories and a dict to map directory to the corresponding labels.

    Args:
        path (str|Path): A path to parent directory of dataset.
        num_classes (int, optional): The number of classes to be
            classified. If None, all of classes are used. Defaults to None.

    Returns:
        tuple[list[str], dict[str, int]]: A list of directory names and a dict to
            map directory names to the corresponding label integers.
    """
    path = Path(path)
    directory_list = list(path.iterdir())

    if num_classes is None or num_classes > len(directory_list):
        num_classes = len(directory_list)

    directory_list = random.sample(directory_list, num_classes)
    label_map = {f: i for i, f in enumerate(directory_list)}

    return directory_list, label_map


def load_imagenet_dataset(
    path: str | Path,
    directory_list: list[str],
    label_map: dict[str, int],
    transform: Any = None,
    train: bool = True,
) -> Dataset:
    """
    Loads imagenet dataset.

    Args:
        path (str|Path): A path to parent directory of dataset.
        directory_list (list[str]): A list of directory names.
        label_map (dict[str, int]): A dict to map directory names on the
            corresponding labels.
        transform (Any, optional): Data transform. Defaults to None.
        train (bool, optional): Whether dataset is for train or not. Defaults
            to True.

    Returns:
        Dataset: A Dataset of imagenet.
    """
    path = Path(path)
    file_list = []
    label_list = []

    for file in directory_list:
        data = list((path / file).iterdir())
        file_list += [str(path / file / f.name) for f in data]
        label_list += [label_map[file]] * len(data)

    dataframe = pd.DataFrame({"id": file_list, "label": label_list})

    if train:
        dataframe = dataframe.sample(frac=1)

    return ImageNetCustomDataset(dataframe, transform=transform)


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


def imagenet_dataset(
    train_path: str | Path,
    valid_path: str | Path,
    train_transforms: Any = None,
    test_transforms: Any = None,
    is_search: bool = True,
) -> tuple[Dataset, Dataset]:
    """
    Loads imagenet dataset for training and validation.

    Args:
        train_path (str|Path): A path to parent directory of training dataset.
        valid_path (str|Path): A path to parent directory of validation dataset.
        train_transforms (Any, optional): Data transform for training dataset. Defaults to None.
        test_transforms (Any, optional): Data transform for validation dataset. Defaults to None.
        is_search (bool, optional): Whether to randomly sample 200 directories. Defaults to True.

    Returns:
        tuple[Dataset, Dataset]: A tuple of training and validation datasets.
    """
    train_path = Path(train_path)
    valid_path = Path(valid_path)
    directory_list = list(train_path.iterdir())

    if is_search:
        directory_list = random.sample(directory_list, 200)

    label_map = {f.name: i for i, f in enumerate(directory_list)}
    train_file_list = []
    train_label_list = []

    for file in directory_list:
        data = list((train_path / file.name).iterdir())
        train_file_list += [str(train_path / file.name / f.name) for f in data]
        train_label_list += [label_map[file.name] for _ in range(len(data))]

    train_data = pd.DataFrame({"id": train_file_list, "label": train_label_list})
    train_data = train_data.sample(frac=1)
    train_dataset = ImageNetCustomDataset(train_data, transform=train_transforms)
    test_file_list = []
    test_label_list = []

    for file in directory_list:
        data = list((valid_path / file.name).iterdir())
        test_file_list += [str(valid_path / file.name / f.name) for f in data]
        test_label_list += [label_map[file.name] for _ in range(len(data))]

    test_data = pd.DataFrame({"id": test_file_list, "label": test_label_list})
    test_dataset = ImageNetCustomDataset(test_data, transform=test_transforms)

    return train_dataset, test_dataset


def _data_transforms_imagenet() -> tuple[transforms.Compose, transforms.Compose]:
    """
    Returns the data transformations for the imagenet dataset.

    Returns:
        Tuple[transforms.Compose, transforms.Compose]: A tuple containing the transformations for the training and
        testing datasets.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, (0.08, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ],
    )
    test_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ],
    )

    return train_transforms, test_transforms


def _data_transforms_cifar() -> tuple[transforms.Compose, transforms.Compose]:
    """
    Get the data transformations for CIFAR dataset.

    Returns:
        Tuple[transforms.Compose, transforms.Compose]: A tuple containing the transformations for training and testing.
    """
    cifar_mean = [0.49139968, 0.48215827, 0.44653124]
    cifar_std = [0.24703233, 0.24348505, 0.26158768]

    train_transforms = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ],
    )

    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cifar_mean, cifar_std),
        ],
    )
    return train_transforms, test_transforms


def drop_path(x: Tensor, drop_prob: float) -> Tensor:
    """
    Drop path in the tensor based on the drop probability.

    Args:
        X (Tensor): The input tensor.
        drop_prob (float): The drop probability.

    Returns:
        Tensor: The tensor after applying drop path.

    Raises:
        AssertionError: If the drop probability is not in the range [0.0, 1.0).
    """
    assert 0.0 <= drop_prob < 1.0, "Drop probability must be in the range [0.0, 1.0)."

    if drop_prob > 0.0:
        keep_prob = 1.0 - drop_prob
        mask = torch.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)

        if x.is_cuda:
            mask = mask.cuda(x.get_device())

        x = x * mask / keep_prob

    return x


def get_search_space_config(search_space_name: str = "proxyless", dataset_name: str = "imagenet") -> Path:
    """
    Get the configuration file path based on the search space and dataset names.

    Args:
        search_space_name (str, optional): The name of the search space. Defaults to "proxyless".
        dataset_name (str, optional): The name of the dataset. Defaults to "imagenet".

    Returns:
        Path: The path to the configuration file.

    Raises:
        ValueError: If the search space or dataset name is invalid.
    """
    valid_search_spaces = ["proxyless", "mnasnet"]
    valid_datasets = ["imagenet", "cifar10", "cifar100"]

    if search_space_name not in valid_search_spaces or dataset_name not in valid_datasets:
        raise ValueError(f"Invalid search space: {search_space_name} or dataset: {dataset_name}")

    # TODO: Replace "config_*_cifar10.yaml" with "config_*_cifar100.yaml" if dataset_name == "cifar100".
    if dataset_name == "cifar100":
        dataset_name = "cifar10"

    return Path(f"./config_{search_space_name}_{dataset_name}.yaml")
