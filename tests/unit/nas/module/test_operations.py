import numpy as np
import pytest
import torch
from torch import nn

import aiaccel
from aiaccel.nas.module.operations import (
    BatchNorm,
    Conv,
    MBConv,
    ReLUOp,
    SEOperation,
    SepConv,
    SkipOperation,
    Zero,
    ch_pad_clip,
    count_params,
)


def test_count_params():
    module = nn.Linear(10, 20)

    param_count = count_params(module)

    assert param_count == 10 * 20 + 20  # weights + biases


def test_ch_pad_clip():
    x = torch.rand(1, 3, 3, 3)

    y = ch_pad_clip(x, 5)
    assert y.shape == (1, 5, 3, 3)
    assert torch.all(y[:, 3:, :, :] == 0)

    y = ch_pad_clip(x, 2)
    assert y.shape == (1, 2, 3, 3)

    y = ch_pad_clip(x, 3)
    assert torch.all(y == x)


def test_conv():
    conv = Conv(max_in_ch=3, max_out_ch=5, k_size=3, stride=1, max_se_ratio=0.5)

    assert conv.max_in_ch == 3
    assert isinstance(conv.conv, nn.Conv2d)
    assert conv.with_bn == True
    assert conv.with_relu == True
    assert conv.groups == 1
    assert conv.k_size == 3

    x = torch.rand(1, 3, 3, 3)
    y = conv.forward(x, filter_size=5, se_ratio=0.5)
    assert y.shape == (1, 5, 3, 3)

    params = conv.get_param_nums(in_ch=3, filter_size=5, se_ratio=0.5)
    assert params == 155

    conv.init_weights()


def test_sepconv():
    sepconv = SepConv(max_in_ch=3, max_out_ch=5, k_size=3, stride=1, max_se_ratio=0.5)

    x = torch.rand(1, 3, 3, 3)
    y = sepconv.forward(x, filter_size=5, se_ratio=0.5)
    assert y.shape == torch.Size([1, 5, 3, 3])

    params = sepconv.get_param_nums(in_ch=3, filter_size=5, se_ratio=0.5)
    assert params == 64

    sepconv.init_weights()


def test_mbconv():
    mbconv = MBConv(max_in_ch=3, max_out_ch=5, k_size=3, stride=1, max_expansion_ratio=1, max_se_ratio=0.5)

    assert mbconv.max_in_ch == 3
    assert mbconv.max_expansion_ratio == 1

    x = torch.rand(1, 3, 3, 3)
    y = mbconv.forward(x, filter_size=5, se_ratio=0.5, expansion_ratio=1)
    assert y.shape == (1, 5, 3, 3)

    params = mbconv.get_param_nums(in_ch=3, filter_size=5, expansion_ratio=1, se_ratio=0.5)
    assert isinstance(params, int)

    mbconv.init_weights()


def test_zero():
    zero = Zero()

    x = torch.rand(1, 3, 3, 3)
    y = zero.forward(x)
    assert torch.all(y == 0)

    params = zero.get_param_nums()
    assert params == 0


def test_reluop():
    reluop = ReLUOp()

    x = torch.rand(1, 3, 3, 3) - 0.5
    y = reluop.forward(x)
    assert torch.all(y >= 0)
    assert torch.all(y[x > 0] == x[x > 0])


def test_batchnorm():
    # Create a BatchNorm instance
    batchnorm = BatchNorm(ch=3)

    # Test the __init__ method
    assert isinstance(batchnorm.batchnorm2d, torch.nn.BatchNorm2d)

    # Test the forward method with a 4D tensor
    x = torch.rand(1, 3, 3, 3)
    y = batchnorm.forward(x)
    assert y.shape == x.shape

    # Test the init_weights method
    batchnorm.init_weights()
    assert torch.all(batchnorm.batchnorm2d.weight == 1)
    assert torch.all(batchnorm.batchnorm2d.bias == 0)


def test_skipoperation():
    # Create a SkipOperation instance
    skipop = SkipOperation(stride=1)

    # Test the __init__ method
    assert isinstance(skipop.max_pool, torch.nn.MaxPool2d)

    # Test the forward method with two 4D tensors
    x = torch.rand(1, 3, 3, 3)
    x_f = torch.rand(1, 3, 3, 3)
    y = skipop.forward(x, x_f)
    assert torch.all(y == x + x_f)


def test_seoperation():
    seop = SEOperation(max_ch=3, se_ch=2, max_se_ratio=0.5)

    # Test the __init__ method
    assert seop.max_ch == 3
    assert seop.max_se_ratio == 0.5
    assert seop.max_se_ch == 2

    # Test the forward method with a 4D tensor
    x = torch.rand(1, 3, 3, 3)
    y = seop.forward(x, se_ch=2, se_ratio=0.5)
    assert y.shape == (1, 3, 3, 3)

    # Test the get_param_nums method
    params = seop.get_param_nums(in_ch=3, se_base_ch=2, se_ratio=0.5)
    assert isinstance(params, int)

    # Test the init_weights method
    seop.init_weights()
    assert torch.all(seop.fc1.weight >= -2) and torch.all(seop.fc1.weight <= 2)
    assert torch.all(seop.fc2.weight >= -2) and torch.all(seop.fc2.weight <= 2)
