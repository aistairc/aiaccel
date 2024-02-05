import pytest
import torch

from aiaccel.nas.mnas_structure_info import MnasNetStructureInfo
from aiaccel.nas.module.operations import (
    BatchNorm,
    Conv,
    MBConv,
    ReLUOp,
    SepConv,
    SkipOperation,
    Zero,
)
from aiaccel.nas.nas_model.proxyless_model import (
    MBConv,
    MnasNetBaseConv,
    MnasNetBlock,
    MnasNetConv,
    MnasNetLayer,
    MnasNetLayerStack,
    MnasNetMBConv,
    MnasNetSearchSpace,
    MnasNetSepConv,
    MnasNetZero,
)


def test_mnasnetbaseconv(mocker):
    # Create a MnasNetBaseConv instance
    mnasnetbaseconv = MnasNetBaseConv(name="MnasNetBaseConv")

    # Mock the not implemented methods
    mocker.patch.object(MnasNetBaseConv, "get_op", return_value=Conv(3, 3, 3, 1, max_se_ratio=0.5))
    mocker.patch.object(MnasNetBaseConv, "select_active_op_sub", return_value=None)
    mocker.patch.object(MnasNetBaseConv, "print_active_op_sub", return_value=None)

    # Test the __init__ method
    assert mnasnetbaseconv.filter_size == -1
    assert mnasnetbaseconv.kernel_size == -1
    assert mnasnetbaseconv.skip_op_str == ""
    assert mnasnetbaseconv.se_ratio == -1
    assert mnasnetbaseconv.prev_filter_size == -1
    assert isinstance(mnasnetbaseconv.candidate_ops, torch.nn.ModuleList)

    # Test the build method
    config = {}
    kwargs = {"max_in_ch": 3, "max_out_ch": 3, "stride": 1, "kernel_size_list": [3, 5, 7], "max_se_ratio": 0.5}
    mnasnetbaseconv.build(config, **kwargs)
    # mnasnetbaseconv.build(config)
    assert mnasnetbaseconv.stride == 1
    assert len(mnasnetbaseconv.candidate_ops) == 3

    # Test the forward method with a 4D tensor
    x = torch.rand(1, 3, 3, 3)
    mnasnetbaseconv.filter_size = 3
    mnasnetbaseconv.se_ratio = 0.5
    mnasnetbaseconv.skip_op_str = "skip"
    mnasnetbaseconv.active_op = mnasnetbaseconv.candidate_ops[0]
    y = mnasnetbaseconv.forward(x)
    assert y.shape == (1, 3, 3, 3)

    # Test the select_active_op method
    structure_info = None
    mnasnetbaseconv.select_active_op(
        structure_info,
        prev_filter_size=3,
        filter_size=3,
        kernel_idx=0,
        kernel_size=3,
        skip_op_str="skip",
        se_ratio=0.5,
    )
    assert mnasnetbaseconv.filter_size == 3
    assert mnasnetbaseconv.kernel_size == 3
    assert mnasnetbaseconv.skip_op_str == "skip"
    assert mnasnetbaseconv.se_ratio == 0.5
    assert mnasnetbaseconv.prev_filter_size == 3
    assert mnasnetbaseconv.active_op == mnasnetbaseconv.candidate_ops[0]

    # Test the enumerate_categorical_variables method
    categorical_variables = mnasnetbaseconv.enumerate_categorical_variables()
    assert categorical_variables == {}

    # Test the enumerate_active_variables method
    active_variables = mnasnetbaseconv.enumerate_active_variables()
    assert active_variables == {}

    # Test the fix_arc method
    mnasnetbaseconv.fix_arc()
    assert len(mnasnetbaseconv.candidate_ops) == 0

    # Test the get_param_num_list method
    params = mnasnetbaseconv.get_param_num_list()
    assert isinstance(params, int)

    # Test the get_hyperparameters method
    max_in_ch, max_out_ch, stride, kernel_size_list, max_se_ratio = mnasnetbaseconv.get_hyperparameters(**kwargs)
    assert max_in_ch == 3
    assert max_out_ch == 3
    assert stride == 1
    assert kernel_size_list == [3, 5, 7]
    assert max_se_ratio == 0.5

    # Test the get_stride method
    stride = mnasnetbaseconv.get_stride()
    assert stride == 1


def test_mnasnetconv(mocker):
    # Create a MnasNetConv instance
    mnasnetconv = MnasNetConv(name="MnasNetConv")

    # Mock the not implemented methods
    mocker.patch.object(MnasNetConv, "select_active_op_sub", return_value=None)
    mocker.patch.object(MnasNetConv, "print_active_op_sub", return_value=None)

    # Test the get_op method
    op = mnasnetconv.get_op(max_in_ch=3, max_out_ch=3, k_size=3, stride=1, max_se_ratio=0.5)
    assert op is not None

    # Test the select_active_op_sub method
    mnasnetconv.select_active_op_sub(structure_info=None)

    # Test the print_active_op_sub method
    mnasnetconv.print_active_op_sub(log_dir=None)


def test_mnasnetsepconv(mocker):
    # Create a MnasNetSepConv instance
    mnasnetsepconv = MnasNetSepConv(name="MnasNetSepConv")

    # Mock the not implemented methods
    mocker.patch.object(MnasNetSepConv, "select_active_op_sub", return_value=None)
    mocker.patch.object(MnasNetSepConv, "print_active_op_sub", return_value=None)

    # Test the get_op method
    op = mnasnetsepconv.get_op(max_in_ch=3, max_out_ch=3, k_size=3, stride=1, max_se_ratio=0.5)
    assert op is not None

    # Test the select_active_op_sub method
    mnasnetsepconv.select_active_op_sub(structure_info=None)

    # Test the print_active_op_sub method
    mnasnetsepconv.print_active_op_sub(log_dir=None)


def test_mnasnetmbconv(mocker):
    # Create a MnasNetMBConv instance
    mnasnetmbconv = MnasNetMBConv(name="MnasNetMBConv")

    # Mock the not implemented methods
    # mocker.patch.object(MnasNetMBConv, "select_active_op_sub", return_value=None)
    # mocker.patch.object(MnasNetMBConv, "print_active_op_sub", return_value=None)

    # Test the get_op method
    op = mnasnetmbconv.get_op(
        max_in_ch=3,
        max_out_ch=3,
        k_size=3,
        stride=1,
        max_expansion_ratio=1,
        max_se_ratio=0.5,
    )
    assert op is not None
    mnasnetmbconv.active_op = op

    # Test the forward method with a 4D tensor
    config = {
        "MnasNetMBConv#expansion_ratio": [0.0, 0.5, 1.0],
    }
    kwargs = {
        "max_in_ch": 3,
        "max_out_ch": 3,
        "stride": 1,
        "kernel_size_list": [3, 5, 7],
        "max_se_ratio": 0.5,
    }
    mnasnetmbconv.build(config, **kwargs)
    x = torch.rand(1, 3, 3, 3)
    y = mnasnetmbconv.forward(x)
    assert y.shape == torch.Size([1, 1, 3, 3])

    # Test the select_active_op_sub method
    structure_info = MnasNetStructureInfo(["MnasNetMBConv#expansion_ratio"])
    mnasnetmbconv.select_active_op_sub(structure_info=structure_info)

    # Test the enumerate_categorical_variables method
    categorical_variables = mnasnetmbconv.enumerate_categorical_variables()
    assert isinstance(categorical_variables, dict)

    # Test the enumerate_active_variables method
    active_variables = mnasnetmbconv.enumerate_active_variables()
    assert isinstance(active_variables, dict)

    # Test the fix_arc method
    mnasnetmbconv.prev_filter_size = 3
    mnasnetmbconv.filter_size = 3
    mnasnetmbconv.kernel_size = 3
    mnasnetmbconv.expansion_ratio = 1
    mnasnetmbconv.fix_arc()

    # Test the print_active_op_sub method
    mnasnetmbconv.print_active_op_sub(log_dir=None)

    # Test the get_param_num_list method
    params = mnasnetmbconv.get_param_num_list()
    assert isinstance(params, int)


def test_mnasnetzero(mocker):
    # Create a MnasNetZero instance
    mnasnetzero = MnasNetZero(name="MnasNetZero")

    # Mock the not implemented methods
    mocker.patch.object(MnasNetZero, "select_active_op_sub", return_value=None)
    mocker.patch.object(MnasNetZero, "print_active_op", return_value=None)

    # Test the get_op method
    op = mnasnetzero.get_op()
    assert op is not None
    mnasnetzero.active_op = op

    # Test the forward method with a 4D tensor
    x = torch.rand(1, 3, 3, 3)
    y = mnasnetzero.forward(x)
    assert y.shape == (1, 3, 3, 3)

    # Test the select_active_op method
    mnasnetzero.select_active_op(structure_info=None, prev_filter_size=3, filter_size=3, skip_op_str="skip")

    # Test the enumerate_categorical_variables method
    categorical_variables = mnasnetzero.enumerate_categorical_variables()
    assert isinstance(categorical_variables, dict)

    # Test the enumerate_active_variables method
    active_variables = mnasnetzero.enumerate_active_variables()
    assert isinstance(active_variables, dict)

    # Test the fix_arc method
    mnasnetzero.fix_arc()

    # Test the print_active_op method
    mnasnetzero.print_active_op(log_dir=None)

    # Test the get_param_num_list method
    params = mnasnetzero.get_param_num_list()
    assert isinstance(params, int)


def test_mnasnetlayer(mocker):
    # Create a MnasNetLayer instance
    mnasnetlayer = MnasNetLayer(name="MnasNetLayer")

    # Test the build method
    config = {
        mnasnetlayer.filter_size_key: [1, 2, 3],
        mnasnetlayer.kernel_size_key: [1, 2, 3],
        mnasnetlayer.conv_op_str_key: ["Conv", "SepConv", "MBConv", "Zero", "MnasNetConv"],
        mnasnetlayer.skip_op_str_key: ["skip"],
        mnasnetlayer.se_ratio_key: [0.0, 0.25, 0.5],
        "MnasNetLayer@MBConv#expansion_ratio": [0.0, 0.5, 1.0],
    }
    mnasnetlayer.build(config, max_in_ch=3, stride=1)
    assert len(mnasnetlayer.candidate_ops) == 4

    # Test the select_active_op method
    structure_info = MnasNetStructureInfo(
        [
            "MnasNetMBConv#expansion_ratio",
            "MnasNetLayer#conv_op_index",
            "MnasNetLayer#filter_size_index",
            "MnasNetLayer#kernel_size_index",
            "MnasNetLayer#skip_op_str",
            "MnasNetLayer#SE_ratio",
        ],
    )
    mnasnetlayer.select_active_op(structure_info, prev_filter_size=3)

    # Test the forward method with a 4D tensor
    x = torch.rand(1, 3, 3, 3)
    y = mnasnetlayer.forward(x)
    assert y.shape == (1, 1, 3, 3)

    # Test the enumerate_categorical_variables method
    categorical_variables = mnasnetlayer.enumerate_categorical_variables()
    assert isinstance(categorical_variables, dict)

    # Test the enumerate_active_variables method
    active_variables = mnasnetlayer.enumerate_active_variables()
    assert isinstance(active_variables, dict)

    # Test the fix_arc method
    mnasnetlayer.fix_arc()

    # Test the print_active_op method
    mnasnetlayer.print_active_op(log_dir=None)

    # Test the get_param_num_list method
    params = mnasnetlayer.get_param_num_list()
    assert isinstance(params, int)

    # Test the get_filter_size method
    filter_size = mnasnetlayer.get_filter_size()
    assert isinstance(filter_size, int)


def test_mnasnetlayerstack(mocker):
    # Create a MnasNetLayerStack instance
    mnasnetlayerstack = MnasNetLayerStack(name="MnasNetLayerStack")

    # Test the build method
    config = {
        mnasnetlayerstack.name + "#max_in_ch": 3,
        mnasnetlayerstack.name + "#stride": 1,
        mnasnetlayerstack.name + "@Layer#filter_size_index": [1, 2, 3],
        mnasnetlayerstack.name + "@Layer#kernel_size_index": [1, 2, 3],
        mnasnetlayerstack.name + "@Layer#conv_op_index": ["Conv", "SepConv", "MBConv", "Zero", "MnasNetConv"],
        mnasnetlayerstack.name + "@Layer#skip_op_str": ["skip"],
        mnasnetlayerstack.name + "@Layer#SE_ratio": [0.0, 0.25, 0.5],
        mnasnetlayerstack.name + "@Layer@MBConv#expansion_ratio": [0.0, 0.5, 1.0],
    }
    mnasnetlayerstack.build(config, layer_num=2)
    assert len(mnasnetlayerstack.layers) == 2

    # Test the select_active_op method
    structure_info = MnasNetStructureInfo(
        [
            mnasnetlayerstack.name + "@Layer#filter_size_index",
            mnasnetlayerstack.name + "@Layer#kernel_size_index",
            mnasnetlayerstack.name + "@Layer#conv_op_index",
            mnasnetlayerstack.name + "@Layer#skip_op_str",
            mnasnetlayerstack.name + "@Layer#SE_ratio",
            mnasnetlayerstack.name + "@Layer@MBConv#expansion_ratio",
        ],
    )
    mnasnetlayerstack.select_active_op(structure_info, prev_filter_size=3)

    # Test the forward method with a 4D tensor
    x = torch.rand(1, 3, 3, 3)
    y = mnasnetlayerstack.forward(x)
    assert y.shape == (1, 1, 3, 3)

    # Test the enumerate_categorical_variables method
    categorical_variables = mnasnetlayerstack.enumerate_categorical_variables()
    assert isinstance(categorical_variables, dict)

    # Test the enumerate_active_variables method
    active_variables = mnasnetlayerstack.enumerate_active_variables()
    assert isinstance(active_variables, dict)

    # Test the fix_arc method
    mnasnetlayerstack.fix_arc()

    # Test the print_active_op method
    mnasnetlayerstack.print_active_op(log_dir=None)

    # Test the get_param_num_list method
    params = mnasnetlayerstack.get_param_num_list()
    assert isinstance(params, list)

    # Test the get_flops_num_list method
    # z = torch.nn.Linear(100, 10)
    flops, x = mnasnetlayerstack.get_flops_num_list(x)
    assert isinstance(flops, list)
    assert x.shape == (1, 1, 3, 3)
