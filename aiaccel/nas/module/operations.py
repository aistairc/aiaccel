from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def count_params(module):
    return np.sum(np.prod(param.size()) for param in module.parameters())


def ch_pad_clip(x: torch.Tensor, trg_ch_size: int) -> torch.Tensor:
    """Pad or Delete channels (ASNG-NAS https://github.com/shirakawas/ASNG-NAS).

    Args:
        x (torch.Tensor): An input tensor.
        trg_ch_size (int): output channels.

    Returns:
        torch.Tensor:
    """
    if trg_ch_size > x.shape[1]:
        # Pad channels to x
        x = F.pad(x, (0, 0, 0, 0, 0, int(trg_ch_size - x.shape[1])), "constant", value=0)
    elif trg_ch_size < x.shape[1]:
        # Delete channels of x
        x = x[:, :trg_ch_size, :, :]
    return x


class Conv(nn.Module):
    """Regular Convolution with SE operation (Conv2d -> BatchNorm2d -> ReLU -> SE operation)

    Args:
        max_in_ch (int): maximum number of input channels.
        max_out_ch (int): maximum number of output channels.
        k_size (int): kernel size.
        stride (int): stride.
        groups (int, optional): At group=1, all inputs are convloved to all outputs.
            At groups = in_channnel, each input channel is convolved with its own set of filters. Defaults to 1.
        bn (bool, optional): whether to apply batchnorm op. Defaults to True.
        relu (bool, optional): whether to apply ReLU op. Defaults to True.
        max_se_ratio (float, optional): maximum value of SE(Squeeze-and-excitation) ratio. Defaults to 0.
    """

    def __init__(
        self,
        max_in_ch: int,
        max_out_ch: int,
        k_size: int,
        stride: int,
        groups: int = 1,
        bn: bool = True,
        relu: bool = True,
        max_se_ratio: int = 0,
    ) -> None:
        super(Conv, self).__init__()
        self.max_in_ch = max_in_ch
        self.conv = nn.Conv2d(
            max_in_ch, max_out_ch, k_size, stride=stride, padding=k_size // 2, groups=int(groups), bias=False
        )

        if max_se_ratio > 0:
            self.se_op = SEOperation(max_out_ch, max_in_ch, max_se_ratio)

        self.with_bn = bn
        self.with_relu = relu

        if bn is True:
            self.bn = BatchNorm(max_out_ch)
        if relu is True:
            self.relu = ReLUOp()

        self.groups = int(groups)
        self.k_size = k_size

    def forward(self, x: torch.Tensor, filter_size: int | None = None, se_ratio: float = 0.0) -> torch.Tensor:
        """Forward function

        Args:
            x (torch.Tensor): An input tensor (batch size, input channel, height, width)
            filter_size (int | None, optional): Filter size (number of output channels). Defaults to None.
            se_ratio (float, optional): value of SE ratio. Deafults to 0.0.

        Returns:
            torch.Tensor: (batch size, output channel, height, width)
        """
        in_ch = x.size(1)
        x = ch_pad_clip(x, self.max_in_ch)
        x = self.conv(x)

        if self.with_bn is True:
            x = self.bn(x)

        if self.with_relu is True:
            x = self.relu(x)

        x = x[:, :filter_size]

        if se_ratio > 0:
            x = self.se_op(x, in_ch, se_ratio=se_ratio)
        x = x[:, :filter_size]
        return x

    def get_param_nums(self, in_ch: int, filter_size: int, se_ratio: float = 0.0) -> int:
        """function to get the number of parameters

        Args:
            in_ch (int): The number of input channels.
            filter_size (int): Filter size.
            se_ratio (float, optional): Value of SE ratio. Defaults to 0.0.

        Returns:
            int: The number of parameters.
        """
        if self.groups == 1:
            params = in_ch * filter_size * self.k_size * self.k_size
        elif self.groups == self.max_in_ch:
            params = in_ch * self.k_size * self.k_size
        if self.with_bn:
            params += 2 * filter_size

        if se_ratio > 0:
            params += self.se_op.get_param_nums(filter_size, in_ch, se_ratio)
        return params

    def init_weights(self) -> None:
        """initialize weights (for debug)."""
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        if hasattr(self, "se_op"):
            self.se_op.init_weights()


class SepConv(nn.Module):
    """Depthwise Separable Convolution with SE operation (DepthwiseConv -> SE operation -> Conv1x1 ->BatchNorm2d)

    Args:
        max_in_ch (int): maximum number of input channels.
        max_out_ch (int) maximum number of output channels.
        k_size (int): kernel size.
        stride (int): stride.
        max_se_ratio (float, optional): maximum value of SE (Squeeze-and-excitation) ratio. Defaults to 0.0.
    """

    def __init__(self, max_in_ch: int, max_out_ch: int, k_size: int, stride: int, max_se_ratio: float = 0.0) -> None:
        super(SepConv, self).__init__()
        self.dw = Conv(max_in_ch, max_in_ch, k_size, stride, groups=max_in_ch)

        if max_se_ratio > 0:
            self.se_op = SEOperation(max_in_ch, max_in_ch, max_se_ratio)

        self.pw_linear = Conv(max_in_ch, max_out_ch, k_size=1, stride=1, bn=True, relu=False)

    def forward(self, x: torch.Tensor, filter_size: int | None = None, se_ratio: float = 0.0) -> torch.Tensor:
        """forward function.

        Args:
            x (torch.Tensor): An input tensor (batch size, in channel, height, width).
            filter_size (int | None, optional): filter size (number of output channels). Defaults to None.
            se_ratio (float): SE ratio.

        Returns:
            torch.Tensor: (batch size, output channel, height, width)
        """
        in_ch = x.shape[1]
        x = self.dw(x, filter_size=in_ch)
        if se_ratio > 0:
            x = self.se_op(x, in_ch, se_ratio=se_ratio)
        x = self.pw_linear(x, filter_size=filter_size)
        return x

    def get_param_nums(self, in_ch: int, filter_size: int, se_ratio: float = 0.0) -> int:
        """function to get the number of parameters

        Args:
            in_ch (int): The number of input channels.
            filter_size (int): Filter size.
            se_ratio (float, optional): Value of SE ratio. Defaults to 0.0.

        Returns
            int: The number of parameters.
        """
        params = self.dw.get_param_nums(in_ch, in_ch)
        if se_ratio > 0:
            params += self.se_op.get_param_nums(in_ch, in_ch, se_ratio)
        params += self.pw_linear.get_param_nums(in_ch, filter_size)
        return params

    def init_weights(self, conv_weight: int, se_weight: int) -> None:
        """initialize weights (for debug)

        Args:
            conv_weight (int): conv weight value to be set.
            se_weight (int): SE weight value to be set.
        """
        self.dw.init_weights(conv_weight, se_weight)
        if hasattr(self, "se_op"):
            print("se_op")
            self.se_op.init_weights(se_weight)

        self.pw_linear.init_weights(conv_weight, se_weight)


class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Convolution (Conv2d -> DepthwiseConv -> SE operation -> Conv1x1 ->BatchNorm2d)

    Args:
        max_in_ch (int): The maximum number of input channels.
        max_out_ch (int): The maximum number of output channels.
        k_size (int): Kernel size.
        stride (int): Stride.
        max_expansion_ratio (int, optional): The maximum value of expansion
            ratio. Defaults to 1
        max_se_ratio (float, optional): The maximum value of SE
            (Squeeze-and-excitation) ratio. Defaults to 0.0
    """

    def __init__(
        self,
        max_in_ch: int,
        max_out_ch: int,
        k_size: int,
        stride: int,
        max_expansion_ratio: int = 1,
        max_se_ratio: float = 0.0,
    ) -> None:
        super(MBConv, self).__init__()
        self.max_in_ch = max_in_ch
        self.max_expansion_ratio = max_expansion_ratio

        self.pw = Conv(max_in_ch, max_in_ch * max_expansion_ratio, k_size=1, stride=1, bn=True, relu=True)
        self.dw = Conv(
            max_in_ch * max_expansion_ratio,
            max_in_ch * max_expansion_ratio,
            k_size,
            stride,
            groups=max_in_ch * max_expansion_ratio,
        )
        if max_se_ratio > 0:
            self.se_op = SEOperation(max_in_ch * max_expansion_ratio, max_in_ch, max_se_ratio)
        self.pw_linear = Conv(max_in_ch * max_expansion_ratio, max_out_ch, k_size=1, stride=1, bn=True, relu=False)

    def forward(
        self, x: torch.Tensor, filter_size: int | None = None, se_ratio: float = 0, expansion_ratio: float = 1
    ) -> torch.Tensor:
        """Forward function.

        Args:
        x (torch.Tensor): An input tensor (batch size, in channel, height, width).
        filter_size (int | None, optional): Filter size (number of output channels). Defaults to None.
        se_ratio (float, optional): Value of SE ratio. Defaults to 0.0
        expansion_ratio (int, optional): Expansion ratio. Defaults to 1.

        Returns:
            torch.Tensor: (batch size, output channel, height, width)
        """
        in_ch = x.shape[1]
        x = self.pw(x, filter_size=in_ch * expansion_ratio)
        x = self.dw(x, filter_size=in_ch * expansion_ratio)
        if se_ratio > 0:
            x = self.se_op(x, in_ch, se_ratio=se_ratio)
        x = self.pw_linear(x, filter_size=filter_size)
        return x

    def get_param_nums(self, in_ch: int, filter_size: int, expansion_ratio: int, se_ratio: float = 0.0) -> int:
        """function to get the number of parameters

        Args:
            in_ch (int): The number of input channels.
            filter_size (int): Filter size.
            se_ratio (float, optional): Value of SE ratio. Defaults to 0.0.

        Returns:
            int: The number of parameters.
        """

        params = self.pw.get_param_nums(in_ch, in_ch * expansion_ratio)
        params += self.dw.get_param_nums(in_ch * expansion_ratio, in_ch * expansion_ratio)
        if se_ratio > 0:
            params += self.se_op.get_param_nums(in_ch * expansion_ratio, in_ch, se_ratio)
        params += self.pw_linear.get_param_nums(in_ch * expansion_ratio, filter_size)
        return params

    def init_weights(self, conv_weight: int, se_weight: int) -> None:
        """Initialize weights (for debug).

        Args:
            conv_weight (int): Conv weight value to be set.
            se_weight (int): SE weight value to be set.
        """
        self.pw.init_weights(conv_weight, se_weight)
        self.dw.init_weights(conv_weight, se_weight)
        if hasattr(self, "se_op"):
            self.se_op.init_weights(se_weight)

        self.pw_linear.init_weights(conv_weight, se_weight)


class Zero(nn.Module):
    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x, filter_size=None, se_ratio=0):
        return torch.zeros_like(x)

    def get_param_nums(self):
        return 0


class ReLUOp(nn.Module):
    """ReLU operation."""

    def __init__(self) -> None:
        super(ReLUOp, self).__init__()
        self.relu = nn.ReLU(
            inplace=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            x (torch.Tensor): An input tensor.

        Returns:
            torch.Tensor: A tensor after applying ReLU.
        """
        return self.relu(x)


class BatchNorm(nn.Module):
    """BatchNorm2d operation."""

    def __init__(self, ch) -> None:
        super(BatchNorm, self).__init__()
        self.batchnorm2d = nn.BatchNorm2d(ch, momentum=0.01, eps=1e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function

        Args:
            x (torch.Tensor): An input tensor.

        Returns:
            torch.Tensor: A tensor after applying BatchNorm.
        """
        x = self.batchnorm2d(x)
        return x

    def init_weights(self):
        nn.init.constant_(self.batchnorm2d.weight, 1)
        nn.init.constant_(self.batchnorm2d.bias, 0)


class SkipOperation(nn.Module):
    """Skip operation.

    If stride=1, identity residual. Otherwise, max pooling.

    Args:
        stride (int): Stride.
    """

    def __init__(self, stride: int) -> None:
        super(SkipOperation, self).__init__()
        self.stride = stride
        self.max_pool = nn.MaxPool2d(stride, stride=stride)

    def forward(self, x, x_f, filter_size=None):
        """Forward function

        Args:
            x (torch.Tensor): An input tensor.
            x_f (torch.Tensor): A tensor after applying conv operations

        Returns:
            torch.Tensor: A tensor after applying skip operation.
        """

        return x + x_f


class SEOperation(nn.Module):
    """SE (Squeeze-and-excitation) operation.

    Args:
        max_ch (int): The maximum number of in/out channels.
        se_ch (int):
        max_se_ratio (float): The maxinum value of SE ratio (0 <= max_se_ratio < 1).
    """

    def __init__(self, max_ch: int, se_ch: int, max_se_ratio: float) -> None:
        super(SEOperation, self).__init__()
        self.max_ch = max_ch
        self.max_se_ratio = max_se_ratio
        self.max_se_ch = se_ch
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(max_ch, int(se_ch * max_se_ratio), bias=True)
        self.fc2 = nn.Linear(int(se_ch * max_se_ratio), max_ch, bias=True)
        self.sig = nn.Sigmoid()
        self.relu = ReLUOp()

    def forward(self, x: torch.tensor, se_ch: int, se_ratio: float = 0.0) -> torch.Tensor:
        """Forward function

        Args:
            x (torch.Tensor): An input tensor (batch size, in/out channel, height, width).
            se_ch (int):
            se_ratio (float, optional): Value of SE ratio. Defaults to 0.0

        Returns
            torch.Tensor: A tensor after applying SE operation (batch size, in/out channel, height, width).
        """
        _, in_out_ch, _, _ = x.size()
        x = ch_pad_clip(x, int(self.max_ch))
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)

        y = self.fc1(y)
        y = y[:, : int(se_ch * se_ratio)]
        y = self.relu(y)
        y = F.pad(y, (0, int(int(self.max_se_ch * self.max_se_ratio) - y.shape[1])), "constant", value=0)
        y = self.fc2(y)
        y = self.sig(y)

        y = y.view(b, c, 1, 1)
        h = x * y.expand_as(x)

        return h

    def get_param_nums(self, in_ch: int, se_base_ch: int, se_ratio: float) -> int:
        """Gets the number of parameters.

        Args:
            in_ch (int):
            ch (int): The number of in/out channels.
            se_ratio (float): Value of SE ratio.

        Returns:
            int: The number of parameters.
        """
        params = in_ch * int(se_base_ch * se_ratio)
        params += int(se_base_ch * se_ratio) * in_ch
        return params

    def init_weights(self) -> None:
        """Initialize weights (for debug)"""
        nn.init.kaiming_uniform_(self.fc1.weight, mode="fan_out", nonlinearity="sigmoid")
        nn.init.kaiming_uniform_(self.fc2.weight, mode="fan_out", nonlinearity="sigmoid")
