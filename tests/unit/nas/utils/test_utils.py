import builtins
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torchvision import transforms

from aiaccel.nas.utils import (
    ImageNetCustomDataset,
    _data_transforms_cifar,
    _data_transforms_imagenet,
    accuracy,
    cross_entropy_with_label_smoothing,
    drop_path,
    get_device,
    get_params,
    get_params2,
    get_params3,
    get_random_dataset_directory,
    get_random_indices,
    get_search_space_config,
    get_yaml_value,
    imagenet_dataset,
    is_valid_yaml_values,
    load_imagenet_dataset,
    load_yaml_file,
    make_categories,
    make_categories2,
    make_observed_values1,
    make_observed_values2,
    make_observed_values3,
)


def test_get_device(mocker):
    mocker.patch("torch.cuda.is_available", return_value=True)
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type == "cuda"

    mocker.patch("torch.cuda.is_available", return_value=False)
    mocker.patch("torch.backends.mps", return_value=True)
    mocker.patch("torch.backends.mps.is_available", return_value=True)
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type == "mps"

    mocker.patch("torch.backends.mps.is_available", return_value=False)
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type == "cpu"


def test_get_yaml_value(mocker):
    mocker.patch("builtins.print")
    input_dict = {"key1": {"key2": "value"}}

    result = get_yaml_value(input_dict, "key1", "key2")
    assert result == "value"

    result = get_yaml_value(input_dict, "key3", "key4")
    assert result is None
    builtins.print.assert_called_once_with("**Set key4.**")


def test_load_yaml_file(mocker):
    mock_yaml_content = """
    key1: value1
    key2: value2
    """
    mocker.patch("yaml.load", return_value={"key1": "value1", "key2": "value2"})
    mocker.patch("pathlib.Path.open", mocker.mock_open(read_data=mock_yaml_content))
    mocker.patch.object(Path, "is_file", return_value=True)
    result = load_yaml_file("test.yaml")
    assert result == {"key1": "value1", "key2": "value2"}

    mocker.patch.object(Path, "is_file", return_value=False)
    with pytest.raises(FileNotFoundError):
        load_yaml_file("test.yaml")


def test_is_valid_yaml_values(mocker):
    # Mock the print function to suppress output during testing
    mocker.patch("builtins.print")

    # Test when all parameters are valid
    valid_params = {
        "k_size_range": [3, 5],
        "base_layer_num": [1, 2],
        "layer_search_range": [1, 2],
        "base_filter_size": [32, 64],
        "filter_size_search_ratio": 0.5,
        "conv_ops": ["Conv", "SepConv", "MBConv"],
        "expansion_ratio": 6,
        "skip_ops": ["no_skip", "skip"],
        "se_ratio": [0.25, 0.5],
        "class_num": 10,
        "strides": [1, 2],
        "first_conv_filter_size": 32,
    }
    assert is_valid_yaml_values(**valid_params) == 1

    invalid_params = valid_params.copy()
    invalid_params["k_size_range"] = None
    assert is_valid_yaml_values(**invalid_params) == 0

    invalid_params = valid_params.copy()
    invalid_params["base_layer_num"].append(3)
    assert is_valid_yaml_values(**invalid_params) == 0

    invalid_params = valid_params.copy()
    invalid_params["conv_ops"].append("InvalidOp")
    assert is_valid_yaml_values(**invalid_params) == 0

    invalid_params = valid_params.copy()
    invalid_params["skip_ops"].append("InvalidOp")
    assert is_valid_yaml_values(**invalid_params) == 0

    invalid_params = valid_params.copy()
    invalid_params["se_ratio"].append(1)
    assert is_valid_yaml_values(**invalid_params) == 0


def test_make_categories():
    category_dic = {
        "Block0#layer_stack_num_index": 1,
        "Block0@LayerStack@Layer#SE_ratio": 0.25,
        "Block0@LayerStack@Layer#filter_size_index": 32,
        "Block0@LayerStack@Layer#kernel_size_index": 3,
        "Block0@LayerStack@Layer#conv_op_index": 2,
        "Block0@LayerStack@Layer@MBConv#expansion_ratio": 6,
        "Block0@LayerStack@Layer#skip_op_str": "no_skip",
    }
    config = {
        "Block0@LayerStack@Layer#conv_op_index": "MBConv",
    }

    expected_result = np.array([1, 168.0, "no_skip"])
    assert np.array_equal(make_categories(category_dic, config), expected_result)

    config["Block0@LayerStack@Layer#conv_op_index"] = "Conv"
    expected_result = np.array([1, 48.0, "no_skip"])
    assert np.array_equal(make_categories(category_dic, config), expected_result)


def test_make_categories2():
    category_dic = {
        "Block0@LayerStack@Layer#SE_ratio": 0.25,
        "Block0@LayerStack@Layer#filter_size_index": 32,
        "Block0@LayerStack@Layer#kernel_size_index": 3,
        "Block0@LayerStack@Layer@MBConv#expansion_ratio": 6,
        "Block1@LayerStack@Layer#SE_ratio": 0.5,
        "Block1@LayerStack@Layer#filter_size_index": 64,
        "Block1@LayerStack@Layer#kernel_size_index": 5,
        "Block1@LayerStack@Layer@MBConv#expansion_ratio": 6,
    }
    config = {
        "Block0@LayerStack@Layer#filter_size_index": [32],
        "Block1@LayerStack@Layer#filter_size_index": [64],
        "Block1@LayerStack#stride": 1,
    }

    expected_result = np.array([144.0])
    assert np.array_equal(make_categories2(category_dic, config), expected_result)

    config["Block1@LayerStack@Layer#filter_size_index"] = [128]
    config["Block1@LayerStack#stride"] = 2
    expected_result = np.array([144.0])
    assert np.array_equal(make_categories2(category_dic, config), expected_result)


def test_make_observed_values1():
    observed_values = np.array([1, 144, 2, 1920])
    config = {
        "Block0@LayerStack@Layer#SE_ratio": [0.25],
        "Block0@LayerStack@Layer#filter_size_index": [32],
        "Block0@LayerStack@Layer#kernel_size_index": [3],
        "Block0@LayerStack@Layer#conv_op_index": ["MBConv"],
        "Block0@LayerStack@Layer@MBConv#expansion_ratio": [6],
        "Block1@LayerStack@Layer#SE_ratio": [0.5],
        "Block1@LayerStack@Layer#filter_size_index": [64],
        "Block1@LayerStack@Layer#kernel_size_index": [5],
        "Block1@LayerStack@Layer#conv_op_index": ["Conv"],
    }
    categories = np.array([1, 144, 2, 1920])

    expected_result = np.array([1, 0, 0, 0, 2, 0, 0])
    assert np.array_equal(make_observed_values1(observed_values, config, categories), expected_result)

    config["Block0@LayerStack@Layer#conv_op_index"] = ["Conv"]
    expected_result = np.array([1, 0, 0, 0, 2, 0, 0])
    assert np.array_equal(make_observed_values1(observed_values, config, categories), expected_result)


def test_make_observed_values2():
    observed_values = np.array([1, 144])
    config = {
        "Block0@LayerStack@Layer#filter_size_index": [32],
        "Block0@LayerStack#stride": 1,
        "Block0@LayerStack@Layer#kernel_size_index": [3],
        "Block0@LayerStack@Layer#conv_op_index": ["MBConv"],
        "Block0@LayerStack@Layer@MBConv#expansion_ratio": [6],
        "Block1@LayerStack@Layer#filter_size_index": [64],
        "Block1@LayerStack#stride": 2,
        "Block1@LayerStack@Layer#kernel_size_index": [5],
        "Block1@LayerStack@Layer#conv_op_index": ["Conv"],
        "Block1@LayerStack@Layer@MBConv#expansion_ratio": [4],
    }
    categories = np.array([1, 144, 2, 1920])

    expected_result = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert np.array_equal(make_observed_values2(observed_values, config, categories), expected_result)

    config["Block0@LayerStack#stride"] = 2
    expected_result = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert np.array_equal(make_observed_values2(observed_values, config, categories), expected_result)


def test_make_observed_values3():
    observed_values = np.array([1, 144])
    config = {
        "Block0@LayerStack@Layer#filter_size_index": [32],
        "Block0@LayerStack#stride": 1,
        "Block0@LayerStack@Layer#kernel_size_index": [3],
        "Block0@LayerStack@Layer#conv_op_index": ["MBConv"],
        "Block0@LayerStack@Layer@MBConv#expansion_ratio": [6],
        "Block1@LayerStack@Layer#filter_size_index": [64],
        "Block1@LayerStack#stride": 2,
        "Block1@LayerStack@Layer#kernel_size_index": [5],
        "Block1@LayerStack@Layer#conv_op_index": ["Conv"],
        "Block1@LayerStack@Layer@MBConv#expansion_ratio": [4],
    }
    categories = np.array([1, 144, 2, 1920])

    expected_result = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert np.array_equal(make_observed_values3(observed_values, config, categories), expected_result)

    config["Block0@LayerStack#stride"] = 2
    expected_result = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert np.array_equal(make_observed_values3(observed_values, config, categories), expected_result)


def test_get_params(mocker):
    nn_model = mocker.Mock()
    structure_info = mocker.Mock()

    nn_model.get_param_num_list.return_value = [[0], [1], [2]]

    mocker.patch("aiaccel.nas.utils.make_observed_values1", return_value=np.array([0, 0, 0]))

    config = {
        "Block0#layer_stack_num_index": 1,
        "Block0@LayerStack@Layer#SE_ratio": [0.25],
        "Block0@LayerStack@Layer#filter_size_index": [32],
        "Block0@LayerStack@Layer#kernel_size_index": [3],
        "Block0@LayerStack@Layer#conv_op_index": ["MBConv"],
        "Block0@LayerStack@Layer@MBConv#expansion_ratio": [6],
        "Block1#layer_stack_num_index": 1,
        "Block1@LayerStack@Layer#SE_ratio": [0.5],
        "Block1@LayerStack@Layer#filter_size_index": [64],
        "Block1@LayerStack@Layer#kernel_size_index": [5],
        "Block1@LayerStack@Layer#conv_op_index": ["Conv"],
    }
    categories = np.array([1, 144, 2, 1920])

    expected_result = np.zeros([len(categories), max(categories)])
    expected_result[0, : categories[0]] = config["Block0#layer_stack_num_index"]
    expected_result[1, : categories[3]] = config["Block1#layer_stack_num_index"]
    assert np.array_equal(get_params(nn_model, structure_info, config, categories), expected_result)


def test_get_params2(mocker):
    nn_model = mocker.Mock()
    structure_info = mocker.Mock()

    nn_model.get_param_num_list.return_value = [[0], [1], [2]]

    mocker.patch("aiaccel.nas.utils.make_observed_values2", return_value=np.array([0, 0, 0]))

    config = {
        "Block0#layer_stack_num_index": 1,
        "Block0@LayerStack#stride": 1,
        "Block0@LayerStack@Layer#SE_ratio": [0.25],
        "Block0@LayerStack@Layer#filter_size_index": [32],
        "Block0@LayerStack@Layer#kernel_size_index": [3],
        "Block0@LayerStack@Layer#conv_op_index": ["MBConv"],
        "Block0@LayerStack@Layer@MBConv#expansion_ratio": [6],
        "Block1#layer_stack_num_index": 2,
        "Block1@LayerStack#stride": 2,
        "Block1@LayerStack@Layer#SE_ratio": [0.5],
        "Block1@LayerStack@Layer#filter_size_index": [64],
        "Block1@LayerStack@Layer#kernel_size_index": [5],
        "Block1@LayerStack@Layer#conv_op_index": ["Conv"],
        "Block1@LayerStack@Layer@MBConv#expansion_ratio": [4],
    }
    categories = np.array([1, 144])

    expected_result = np.zeros([len(categories), max(categories)])
    expected_result[0, : categories[0]] = config["Block0#layer_stack_num_index"]
    expected_result[1, : categories[1]] = config["Block1#layer_stack_num_index"]
    assert np.array_equal(get_params2(nn_model, structure_info, config, categories), expected_result)


def test_get_params3(mocker):
    nn_model = mocker.Mock()
    structure_info = mocker.Mock()

    nn_model.get_param_num_list.side_effect = lambda: [[0], [1], [2]]

    mocker.patch("aiaccel.nas.utils.make_observed_values3", return_value=np.array([0, 0, 0]))

    config = {
        "Block0#layer_stack_num_index": 1,
        "Block0@LayerStack#stride": 1,
        "Block0@LayerStack@Layer#SE_ratio": [0.25],
        "Block0@LayerStack@Layer#filter_size_index": [32],
        "Block0@LayerStack@Layer#kernel_size_index": [3],
        "Block0@LayerStack@Layer#conv_op_index": ["MBConv"],
        "Block0@LayerStack@Layer@MBConv#expansion_ratio": [6],
        "Block1#layer_stack_num_index": 2,
        "Block1@LayerStack#stride": 2,
        "Block1@LayerStack@Layer#SE_ratio": [0.5],
        "Block1@LayerStack@Layer#filter_size_index": [64],
        "Block1@LayerStack@Layer#kernel_size_index": [5],
        "Block1@LayerStack@Layer#conv_op_index": ["Conv"],
        "Block1@LayerStack@Layer@MBConv#expansion_ratio": [4],
    }
    categories = np.array([1, 144])

    expected_result = np.zeros([len(categories), max(categories)])
    expected_result[0, : categories[0]] = config["Block0#layer_stack_num_index"]
    expected_result[1, : categories[1]] = config["Block1#layer_stack_num_index"]
    assert np.array_equal(get_params3(nn_model, structure_info, config, categories), expected_result)


def test_accuracy():
    output = torch.tensor([[0.1, 0.2, 0.7], [0.5, 0.2, 0.3]])
    target = torch.tensor([2, 0])

    expected_result = [torch.tensor([100.0])]
    assert torch.allclose(torch.stack(accuracy(output, target)), torch.stack(expected_result), atol=1e-4)

    expected_result = [torch.tensor([100.0]), torch.tensor([100.0])]
    assert torch.allclose(torch.stack(accuracy(output, target, topk=(1, 2))), torch.stack(expected_result), atol=1e-4)


def test_cross_entropy_with_label_smoothing():
    pred = torch.tensor([[0.1, 0.2, 0.7], [0.5, 0.2, 0.3]])
    target = torch.tensor([2, 0])

    expected_result = torch.tensor(0.8806)
    assert torch.allclose(
        cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.1),
        expected_result,
        atol=1e-4,
    )

    expected_result = torch.tensor(0.9072)
    assert torch.allclose(
        cross_entropy_with_label_smoothing(pred, target, label_smoothing=0.2),
        expected_result,
        atol=1e-4,
    )


def test_ImageNetCustomDataset(mocker):
    mocker.patch("pathlib.Path.open", mocker.mock_open(read_data="Fake data"))
    mocker.patch("PIL.Image.open", return_value=Image.Image())
    mocker.patch("PIL.Image.Image.convert", return_value="RGB image")

    dataframe = pd.DataFrame(
        {
            "id": ["path/to/image1", "path/to/image2"],
            "label": [0, 1],
        },
    )

    dataset = ImageNetCustomDataset(dataframe)
    assert len(dataset) == 2

    img, label = dataset[0]
    assert img == "RGB image"
    assert label == 0


def test_get_random_dataset_directory(mocker):
    mocker.patch("pathlib.Path.iterdir", return_value=["dir1", "dir2", "dir3", "dir4", "dir5"])

    directory_list, label_map = get_random_dataset_directory("path/to/dataset", num_classes=None)
    assert len(directory_list) == 5
    assert len(label_map) == 5

    directory_list, label_map = get_random_dataset_directory("path/to/dataset", num_classes=3)
    assert len(directory_list) == 3
    assert len(label_map) == 3


def test_load_imagenet_dataset(mocker):
    mocker.patch("pathlib.Path.iterdir", return_value=[Path(), Path(), Path()])
    mocker.patch("aiaccel.nas.utils.ImageNetCustomDataset", return_value="Mocked dataset")

    directory_list = ["dir1", "dir2"]
    label_map = {"dir1": 0, "dir2": 1}

    dataset = load_imagenet_dataset("path/to/dataset", directory_list, label_map, train=True)
    assert len(dataset) == 6

    dataset = load_imagenet_dataset("path/to/dataset", directory_list, label_map, train=False)
    assert len(dataset) == 6


def test_get_random_indices():
    subset_indices, remained_subset_indices = get_random_indices(10, 3)
    assert len(subset_indices) == 3
    assert len(remained_subset_indices) == 7
    assert len(set(subset_indices).intersection(set(remained_subset_indices))) == 0

    subset_indices, remained_subset_indices = get_random_indices(100, 50)
    assert len(subset_indices) == 50
    assert len(remained_subset_indices) == 50
    assert len(set(subset_indices).intersection(set(remained_subset_indices))) == 0


def test_imagenet_dataset(mocker):
    mocker.patch("pathlib.Path.iterdir", return_value=[Path() for i in range(300)])

    mocker.patch("aiaccel.nas.utils.ImageNetCustomDataset", return_value="Mocked dataset")

    train_dataset, test_dataset = imagenet_dataset("path/to/train", "path/to/test", is_search=True)
    assert len(train_dataset) == 60000
    assert len(test_dataset) == 60000

    train_dataset, test_dataset = imagenet_dataset("path/to/train", "path/to/test", is_search=False)
    assert len(train_dataset) == 90000
    assert len(test_dataset) == 90000


def test_data_transforms_imagenet(mocker):
    mocker.patch("torchvision.transforms.Compose", return_value="Mocked transform")

    train_transforms, test_transforms = _data_transforms_imagenet()
    assert train_transforms == "Mocked transform"
    assert test_transforms == "Mocked transform"


def test_data_transforms_cifar(mocker):
    mocker.patch("torchvision.transforms.Compose", return_value="Mocked transform")

    train_transforms, test_transforms = _data_transforms_cifar()
    assert train_transforms == "Mocked transform"
    assert test_transforms == "Mocked transform"


def test_drop_path(mocker):
    tensor_mock = mocker.Mock(spec=torch.Tensor)
    tensor_mock.is_cuda = False
    tensor_mock.get_device.return_value = 0

    x = drop_path(tensor_mock, 0.0)
    assert x == tensor_mock

    tensor = torch.Tensor([1])
    x = drop_path(tensor, 0.5)
    assert x == torch.Tensor([[[[0.0]]]])

    with pytest.raises(AssertionError):
        x = drop_path(tensor_mock, 1.0)


def test_get_search_space_config():
    config_path = get_search_space_config("proxyless", "imagenet")
    assert config_path == Path("./config_proxyless_imagenet.yaml")

    config_path = get_search_space_config("mnasnet", "cifar10")
    assert config_path == Path("./config_mnasnet_cifar10.yaml")

    config_path = get_search_space_config("proxyless", "cifar100")
    assert config_path == Path("./config_proxyless_cifar10.yaml")

    with pytest.raises(ValueError):
        config_path = get_search_space_config("invalid", "imagenet")

    with pytest.raises(ValueError):
        config_path = get_search_space_config("proxyless", "invalid")
