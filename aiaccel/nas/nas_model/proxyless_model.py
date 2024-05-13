from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from ptflops import get_model_complexity_info
from torch import nn

from aiaccel.nas.module.nas_module import NASModule
from aiaccel.nas.module.operations import (
    BatchNorm,
    Conv,
    MBConv,
    ReLUOp,
    SepConv,
    SkipOperation,
    Zero,
)

if TYPE_CHECKING:
    from aiaccel.nas.mnas_structure_info import MnasNetStructureInfo

INDENT = "    "


class MnasNetBaseConv(NASModule):
    """Subclass (MnasNetXConv) set supernet or subnet from Xconv operation class (Conv/SepConv/MBConv).

    There are no categorical variables belonging to this class.

    Args:
        name (str): Class name.
    """

    def __init__(self, name: str) -> None:
        super(__class__, self).__init__(name)

        self.filter_size = -1
        self.kernel_size = -1
        self.skip_op_str = ""
        self.se_ratio = -1
        self.prev_filter_size = -1

        self.candidate_ops = nn.ModuleList([])

    def build(self, config: dict[str, Any], **kwargs) -> None:
        """Create supernet based on config file.

        Args:
            config (dict): A dict type config that can be created using util function.
            max_in_ch (int): The maximum number of input channels.
            max_out_ch (int): The maximum number of output channels.
            stride (int): Stride.
            kernel_size_list (list[int]): A list of kernel size.
            max_se_ratio (int): The maximum value of SE (Squeeze-and-excitation) ratio.
        """

        max_in_ch, max_out_ch, stride, kernel_size_list, max_se_ratio = self.get_hyperparameters(**kwargs)
        self.stride = stride
        self.skip_op = SkipOperation(stride)

        for k_size in kernel_size_list:
            self.candidate_ops.append(self.get_op(max_in_ch, max_out_ch, k_size, stride, max_se_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function

        Args:
            x (torch.Tensor): An input tensor (batch size, input channel, height, width).

        Returns
            torch.Tensor: A tensor after operation (batch size, output channel, height, width).
        """
        x_f = self.active_op(x, filter_size=self.filter_size, se_ratio=self.se_ratio)

        if self.skip_op_str == "skip" and x.size()[1] == self.filter_size and self.stride == 1:
            x_f = self.skip_op(x, x_f, filter_size=self.filter_size)

        return x_f

    def select_active_op(self, structure_info: MnasNetStructureInfo, prev_filter_size: int, **kwargs) -> None:
        """Set subnet based on structure_info.

        Args:
            structure_info (MnasNetStructureInfo): Categorical keys/variables.
            prev_filter_size (int): Filter size of the previous LAYER.
        """
        self.filter_size = kwargs["filter_size"]
        kernel_idx = kwargs["kernel_idx"]
        self.kernel_size = kwargs["kernel_size"]
        self.skip_op_str = kwargs["skip_op_str"]
        self.se_ratio = kwargs["se_ratio"]

        self.prev_filter_size = prev_filter_size

        self.active_op = self.candidate_ops[kernel_idx]

        self.select_active_op_sub(structure_info)

    def enumerate_categorical_variables(self):
        # This class does not have any categorical variables.
        return {}

    def enumerate_active_variables(self):
        # This class does not have any categorical variables.
        return {}

    def fix_arc(self, **kwargs):
        self.candidate_ops = nn.ModuleList({})

        stride = self.get_stride()
        self.active_op = self.get_op(self.prev_filter_size, self.filter_size, self.kernel_size, stride, self.se_ratio)

    def print_active_op(self, log_dir: Path | None = None):
        params = self.get_param_num_list()

        params_str = f"params: {params:>10} "

        name_str_list = self.name.split("@")

        conv_str = f"{name_str_list[-1]:<7} "

        kernel_size_str = f"kernel_size: {self.kernel_size}  "
        filter_size_str = f"filter_size: {self.filter_size}  "
        stride_str = f"stride: {self.get_stride()} "
        skip_op_str = f"skip_op: {self.skip_op_str}  "
        se_ratio_str = f"se_ratio: {self.se_ratio}  "

        # print(INDENT * 2 + conv_str + params_str)
        # print(INDENT * 4 + kernel_size_str + filter_size_str + stride_str + skip_op_str + se_ratio_str)

        if log_dir is not None:
            with Path.open(log_dir / "architecture.txt", "a") as o:
                o.write(INDENT * 2 + conv_str + params_str + "\n")
                o.write(INDENT * 4 + kernel_size_str + filter_size_str + stride_str + skip_op_str + se_ratio_str + "\n")

        self.print_active_op_sub(log_dir)

    def get_param_num_list(self):
        return self.active_op.get_param_nums(self.prev_filter_size, self.filter_size, se_ratio=self.se_ratio)

    def get_hyperparameters(self, **kwargs):
        max_in_ch = kwargs["max_in_ch"]
        max_out_ch = kwargs["max_out_ch"]
        stride = kwargs["stride"]
        kernel_size_list = kwargs["kernel_size_list"]
        max_se_ratio = kwargs["max_se_ratio"]

        return max_in_ch, max_out_ch, stride, kernel_size_list, max_se_ratio

    def get_stride(self):
        return self.skip_op.stride

    def get_op(self):
        raise NotImplementedError

    def select_active_op_sub(self, structure_info):
        raise NotImplementedError

    def print_active_op_sub(self, log_dir=None):
        raise NotImplementedError


class MnasNetConv(MnasNetBaseConv):
    """Set supernet or subnet for operation class Conv.

    This class does not contain categorical variables.
    """

    def get_op(self, max_in_ch, max_out_ch, k_size, stride, max_se_ratio):
        return Conv(max_in_ch, max_out_ch, k_size, stride, max_se_ratio=max_se_ratio)

    def select_active_op_sub(self, structure_info):
        pass

    def print_active_op_sub(self, log_dir=None):
        pass


class MnasNetSepConv(MnasNetBaseConv):
    """Set supernet or subnet for operation class SepConv.

    This class does not contain categorical variables.
    """

    def get_op(self, max_in_ch, max_out_ch, k_size, stride, max_se_ratio):
        return SepConv(max_in_ch, max_out_ch, k_size, stride, max_se_ratio=max_se_ratio)

    def select_active_op_sub(self, structure_info):
        pass

    def print_active_op_sub(self, log_dir=None):
        pass


class MnasNetMBConv(MnasNetBaseConv):
    """Set supernet or subnet for operation class MBConv.

    categorical variables: expansion ratio.
    """

    def __init__(self, name):
        super(__class__, self).__init__(name)

        self.expansion_ratio = -1

        self.expansion_ratio_key = self.name + "#expansion_ratio"
        self.expansion_ratio_list = []

    def get_op(self, max_in_ch, max_out_ch, k_size, stride, max_expansion_ratio, max_se_ratio):
        return MBConv(
            max_in_ch,
            max_out_ch,
            k_size,
            stride,
            max_expansion_ratio=max_expansion_ratio,
            max_se_ratio=max_se_ratio,
        )

    def build(self, config, **kwargs):
        """Create supernet based on config file.

        Args:
            config (dict[str, Any]): A dict type config that can be created using util function.
            max_in_ch (int): The maximum number of input channels.
            max_out_ch (int): The maximum number of output channels.
            stride (int): stride.
            kernel_size_list (list[int]): A list of kernel size.
            max_se_ratio (int): The maximum value of SE (Squeeze-and-excitation) ratio.
        """

        max_in_ch, max_out_ch, stride, kernel_size_list, max_se_ratio = self.get_hyperparameters(**kwargs)
        self.stride = stride
        self.skip_op = SkipOperation(stride)

        self.expansion_ratio_list = config[self.expansion_ratio_key]
        max_expansion_ratio = int(np.max(self.expansion_ratio_list))

        for k_size in kernel_size_list:
            self.candidate_ops.append(
                self.get_op(max_in_ch, max_out_ch, k_size, stride, max_expansion_ratio, max_se_ratio),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function

        Args:
            x (torch.Tensor): An input tensor (batch size, input channel, height, width).

        Returns
            torch.Tensor: A tensor after operation (batch size, output channel, height, width).
        """
        x_f = self.active_op(
            x,
            filter_size=self.filter_size,
            se_ratio=self.se_ratio,
            expansion_ratio=self.expansion_ratio,
        )

        if self.skip_op_str == "skip" and x.size()[1] == self.filter_size and self.stride == 1:
            x_f = self.skip_op(x, x_f, filter_size=self.filter_size)

        return x_f

    def select_active_op_sub(self, structure_info):
        self.expansion_ratio = self.expansion_ratio_list[structure_info(self.expansion_ratio_key)]

    def enumerate_categorical_variables(self):
        # categorical key/variable(expansion_ratio)
        return {
            self.expansion_ratio_key: len(self.expansion_ratio_list),
        }

    def enumerate_active_variables(self):
        # active key/variable(expansion_ratio)
        return {
            self.expansion_ratio_key: (self.expansion_ratio_list).index(self.expansion_ratio),
        }

    def fix_arc(self, **kwargs):
        self.candidate_ops = nn.ModuleList({})
        stride = self.get_stride()
        self.active_op = self.get_op(
            self.prev_filter_size,
            self.filter_size,
            self.kernel_size,
            stride,
            self.expansion_ratio,
            self.se_ratio,
        )

    def print_active_op_sub(self, log_dir: Path | None = None):
        expansion_ratio_str = f"expansion_ratio: {self.expansion_ratio}  "

        # print(INDENT * 4 + expansion_ratio_str)
        if log_dir is not None:
            with Path.open(log_dir / "architecture.txt", "a") as o:
                o.write(INDENT * 4 + expansion_ratio_str + "\n")

    def get_param_num_list(self):
        return self.active_op.get_param_nums(
            self.prev_filter_size,
            self.filter_size,
            self.expansion_ratio,
            self.se_ratio,
        )


class MnasNetZero(MnasNetBaseConv):
    def __init__(self, name):
        super(__class__, self).__init__(name)

    def get_op(self):
        return Zero()

    def build(self, config, **kwargs):
        max_in_ch, max_out_ch, stride, kernel_size_list, max_se_ratio = self.get_hyperparameters(**kwargs)
        self.skip_op = SkipOperation(stride)

    def forward(self, x):
        x_f = self.active_op(x)

        if self.skip_op_str == "skip" and x.size()[1] == self.filter_size:
            x_f = self.skip_op(x, x_f, filter_size=self.filter_size)

        return x_f

    def select_active_op_sub(self, structure_info):
        pass

    def select_active_op(self, structure_info, prev_filter_size, **kwargs):
        self.filter_size = kwargs["filter_size"]
        self.skip_op_str = kwargs["skip_op_str"]

        self.prev_filter_size = prev_filter_size
        self.active_op = Zero()

    def enumerate_categorical_variables(self):
        # categorical key/variable(expansion_ratio)
        return {}

    def enumerate_active_variables(self):
        # active key/variable(expansion_ratio)
        return {}

    def fix_arc(self, **kwargs):
        pass

    def print_active_op(self, log_dir: Path | None = None):
        params = self.get_param_num_list()
        params_str = f"params: {params:>10} "
        name_str_list = self.name.split("@")
        conv_str = f"{name_str_list[-1]:<7} "

        filter_size_str = f"filter_size: {self.filter_size}  "
        stride_str = f"stride: {self.get_stride()} "
        skip_op_str = f"skip_op: {self.skip_op_str}  "

        # print(INDENT * 2 + conv_str + params_str)
        # print(INDENT * 4 + filter_size_str + stride_str + skip_op_str)

        if log_dir is not None:
            with Path.open(log_dir / "architecture.txt", "a") as o:
                o.write(INDENT * 2 + conv_str + params_str + "\n")
                o.write(INDENT * 4 + filter_size_str + stride_str + skip_op_str + "\n")

    def get_param_num_list(self):
        return self.active_op.get_param_nums()


class MnasNetLayer(NASModule):
    """Layer is composed of conv (MnasNetXConv).

    Categorical variables: filter_size, kernel size, conv op, skip op, SE ratio.
    """

    def __init__(self, name):
        super(__class__, self).__init__(name)

        # filter size
        self.filter_size_key = self.name + "#filter_size_index"
        self.filter_size_list = []

        # kernel size
        self.kernel_size_key = self.name + "#kernel_size_index"
        self.kernel_size_list = []

        # conv op
        self.conv_op_str_key = self.name + "#conv_op_index"
        self.conv_op_str_list = []

        # skip op
        self.skip_op_str_key = self.name + "#skip_op_str"
        self.skip_op_str_list = []

        # SE ratio
        self.se_ratio_key = self.name + "#SE_ratio"
        self.se_ratio_list = []

        # choice of operations
        self.candidate_ops = nn.ModuleList([])

    def build(self, config: dict[str, Any], **kwargs) -> None:
        """Create supernet based on config file.

        Args:
            config (dict[str, Any]): A dict type config that can be created using util function
            max_in_ch (int): The maximum number of input channels.
            stride (int): Stride.
        """
        self.filter_size_list = config[self.filter_size_key]
        self.kernel_size_list = config[self.kernel_size_key]
        self.conv_op_str_list = config[self.conv_op_str_key]
        self.skip_op_str_list = config[self.skip_op_str_key]
        self.se_ratio_list = config[self.se_ratio_key]

        max_in_ch = kwargs["max_in_ch"]
        max_out_ch = int(np.max(self.filter_size_list))

        stride = kwargs["stride"]
        max_se_ratio = float(np.max(self.se_ratio_list))

        for conv_op_str in self.conv_op_str_list:
            name = self.name + "@" + conv_op_str

            if conv_op_str == "Conv":
                conv_op = MnasNetConv(name)
            elif conv_op_str == "SepConv":
                conv_op = MnasNetSepConv(name)
            elif conv_op_str == "MBConv":
                conv_op = MnasNetMBConv(name)
            elif conv_op_str == "Zero":
                conv_op = MnasNetZero(name)
            else:
                return

            conv_op.build(
                config,
                kernel_size_list=self.kernel_size_list,
                max_in_ch=max_in_ch,
                max_out_ch=max_out_ch,
                stride=stride,
                max_se_ratio=max_se_ratio,
            )
            self.candidate_ops.append(conv_op)

    def select_active_op(self, structure_info: MnasNetStructureInfo, prev_filter_size: int, **kwargs):
        """Set subnet based on structure_info.

        Args:
            structure_info (MnasNetStructureInfo): Categorical keys/variables.
            prev_filter_size (int): Filter size of the previous LAYER.
        """
        # set subnet(MnasNetXConv) by categorical variables(filter size, skip op, SE ratio, conv op)
        filter_size = self.filter_size_list[structure_info(self.filter_size_key)]
        kernel_size = self.kernel_size_list[structure_info(self.kernel_size_key)]
        skip_op_str = self.skip_op_str_list[structure_info(self.skip_op_str_key)]
        se_ratio = self.se_ratio_list[structure_info(self.se_ratio_key)]

        kernel_idx = self.kernel_size_list.index(kernel_size)

        self.active_op = self.candidate_ops[structure_info(self.conv_op_str_key)]
        # recursively setting
        self.active_op.select_active_op(
            structure_info,
            prev_filter_size,
            filter_size=filter_size,
            kernel_idx=kernel_idx,
            kernel_size=kernel_size,
            skip_op_str=skip_op_str,
            se_ratio=se_ratio,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function

        Args:
            x (torch.Tensor): An input tensor (batch size, input channel, height, width).

        Returns
            torch.Tensor: A tensor after operation (batch size, output channel, height, width).
        """
        return self.active_op(x)

    def enumerate_categorical_variables(self):
        # categorical keys/variables(filter_size, kernel_size, conv_op_str, skip_op_str, se_ratio)
        categorical_variables = {
            self.filter_size_key: len(self.filter_size_list),
            self.kernel_size_key: len(self.kernel_size_list),
            self.conv_op_str_key: len(self.conv_op_str_list),
            self.skip_op_str_key: len(self.skip_op_str_list),
            self.se_ratio_key: len(self.se_ratio_list),
        }

        # recursively getting
        for conv_op in self.candidate_ops:
            sub_cat_vars = conv_op.enumerate_categorical_variables()
            for key, var in sub_cat_vars.items():
                if key not in categorical_variables:
                    categorical_variables[key] = var

        return categorical_variables

    def enumerate_active_variables(self):
        # categorical keys/variables(filter_size, kernel_size, conv_op_str, skip_op_str, se_ratio)
        # conv_op_str = self.active_op.name.split("@")[-1]
        conv_op_str = self.active_op.__class__.__name__
        active_variables = {
            self.filter_size_key: (self.filter_size_list).index(self.active_op.filter_size),
            self.kernel_size_key: (self.kernel_size_list).index(self.active_op.kernel_size),
            self.conv_op_str_key: (self.conv_op_str_list).index(conv_op_str),
            self.skip_op_str_key: (self.skip_op_str_list).index(self.active_op.skip_op_str),
            self.se_ratio_key: (self.se_ratio_list).index(self.active_op.se_ratio),
        }

        # recursively getting
        sub_cat_vars = self.active_op.enumerate_active_variables()
        for key, var in sub_cat_vars.items():
            if key not in active_variables:
                active_variables[key] = var

        return active_variables

    def fix_arc(self, **kwargs):
        self.candidate_ops = nn.ModuleList([])

        self.active_op.fix_arc()

    def print_active_op(self, log_dir=None):
        self.active_op.print_active_op(log_dir)

    def get_param_num_list(self):
        return self.active_op.get_param_num_list()

    def get_filter_size(self):
        return self.active_op.filter_size


class MnasNetLayerStack(NASModule):
    """LayerStack is composed of layers (MnasNetLayer).

    There are no categorical variables belonging to this class.

    Args:
        name (str): Class name.
    """

    def __init__(self, name: str) -> None:
        super(__class__, self).__init__(name)

        # List of MnasNetLayers
        self.layers = nn.ModuleList([])

    def build(self, config, **kwargs):
        """Create supernet based on config file.

        Args:
            config (dict[str, Any]): A dict type config that can be created using util function.
            layer_num (int): The number of layer stack.
        """
        layer_num = kwargs["layer_num"]

        max_in_ch = config[self.name + "#max_in_ch"]
        stride = config[self.name + "#stride"]

        max_out_ch = -1

        for layer_idx in range(layer_num):
            name = self.name + "@Layer"
            layer = MnasNetLayer(name)
            if layer_idx == 0:
                layer.build(config, max_in_ch=max_in_ch, stride=stride)
                max_out_ch = int(np.max(layer.filter_size_list))
            else:
                layer.build(config, max_in_ch=max_out_ch, stride=1)

            self.layers.append(layer)

    def select_active_op(self, structure_info, prev_filter_size, **kwargs):
        """Set subnet based on structure_info.

        Args:
            structure_info (MnasNetStructureInfo): Categorical keys/variables.
            prev_filter_size (int): Filter size of the previous block.
        """

        # This class does not have any categorical variables.

        # recursively setting
        for layer in self.layers:
            layer.select_active_op(structure_info, prev_filter_size)
            prev_filter_size = layer.get_filter_size()

        self.filter_size = prev_filter_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function

        Args:
            x (torch.Tensor): An input tensor (batch size, input channel, height, width).

        Returns
            torch.Tensor: A tensor after operation (batch size, output channel, height, width).
        """
        for layer in self.layers:
            x = layer.active_op(x)

        return x

    def enumerate_categorical_variables(self):
        # This class does not have any categorical variables.
        categorical_variables = {}

        # recursively getting
        for layer in self.layers:
            sub_cat_vars = layer.enumerate_categorical_variables()
            for key, var in sub_cat_vars.items():
                if key not in categorical_variables:
                    categorical_variables[key] = var

        return categorical_variables

    def enumerate_active_variables(self):
        # This class does not have any categorical variables.
        active_variables = {}

        # recursively getting
        for layer in self.layers:
            sub_cat_vars = layer.enumerate_active_variables()
            for key, var in sub_cat_vars.items():
                if key not in active_variables:
                    active_variables[key] = var

        return active_variables

    def fix_arc(self, **kwargs):
        for layer in self.layers:
            layer.fix_arc()

    def print_active_op(self, log_dir: Path | None = None):
        layer_num = len(self.layers)
        # print(INDENT + "LayerNum=" + str(layer_num))
        if log_dir is not None:
            with Path.open(log_dir / "architecture.txt", "a") as o:
                o.write(INDENT + "LayerNum=" + str(layer_num) + "\n")

        # recursively printing
        for layer in self.layers:
            layer.print_active_op(log_dir=log_dir)

    def get_param_num_list(self):
        params_list = []
        for layer in self.layers:
            params = layer.get_param_num_list()
            params_list.append(params)

        return params_list

    def get_flops_num_list(self, x):
        flops_list = []
        for layer in self.layers:
            flop, _ = get_model_complexity_info(
                layer,
                tuple(x.size()[1:]),
                as_strings=False,
                print_per_layer_stat=False,
                verbose=False,
            )
            flops_list.append(x)
            x = layer(x)

        return flops_list, x


class MnasNetBlock(NASModule):
    """Block is composed of layer stack (MnasNetLayerStack).

    categorical variables: layer stack num

    Args:
        name (str): Class name.
    """

    def __init__(self, name: str) -> None:
        super(__class__, self).__init__(name)

        # categorical variables key and list
        self.layer_stack_num_key = name + "#layer_stack_num_index"
        self.layer_stack_num_list = []  # buildで設定

        # choice of operations
        self.candidate_ops = nn.ModuleList([])

    def build(self, config: dict[str, Any], **kwargs) -> None:
        """Create supernet based on config file.

        Args:
            config (dict): A dict type config that can be created using util function.
        """
        self.layer_stack_num_list = config[self.layer_stack_num_key]

        for layer_num in self.layer_stack_num_list:
            name = self.name + "@LayerStack"
            layer_stack = MnasNetLayerStack(name)
            layer_stack.build(config, layer_num=layer_num)
            self.candidate_ops.append(layer_stack)

    def select_active_op(self, structure_info, prev_filter_size, **kwargs):
        """Set subnet based on structure_info.

        Args:
            structure_info (MnasNetStructureInfo): Categorical keys/variables.
            prev_filter_size (int): Filter size of the previous block.
        """
        # set subnet(LayerStack) by categorical variables(layer_stack_num)
        self.active_op = self.candidate_ops[structure_info(self.layer_stack_num_key)]
        # recursively setting
        self.active_op.select_active_op(structure_info, prev_filter_size)

        self.filter_size = self.active_op.filter_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function

        Args:
            x (torch.Tensor): An input tensor (batch size, input channel, height, width).

        Returns
            torch.Tensor: A tensor after operation (batch size, output channel, height, width).
        """
        return self.active_op(x)

    def enumerate_categorical_variables(self):
        # categorical key/variable(layer_num)
        categorical_variables = {self.layer_stack_num_key: len(self.layer_stack_num_list)}

        # recursively getting
        for layer_stack in self.candidate_ops:
            sub_cat_vars = layer_stack.enumerate_categorical_variables()
            for key, var in sub_cat_vars.items():
                if key not in categorical_variables:
                    categorical_variables[key] = var

        return categorical_variables

    def enumerate_active_variables(self):
        # active key/variable(layer_num)
        layer_stack_num = len(self.active_op.layers)
        active_variables = {self.layer_stack_num_key: self.layer_stack_num_list.index(layer_stack_num)}

        # recursively getting
        sub_cat_vars = self.active_op.enumerate_active_variables()
        for key, var in sub_cat_vars.items():
            if key not in active_variables:
                active_variables[key] = var

        return active_variables

    def fix_arc(self, **kwargs):
        self.candidate_ops = nn.ModuleList([])

        self.active_op.fix_arc()

    def print_active_op(self, log_dir: Path | None = None):
        # print("[" + self.name + "]")
        if log_dir is not None:
            with Path.open(log_dir / "architecture.txt", "a") as o:
                o.write("[" + self.name + "]" + "\n")

        # recursively printing
        self.active_op.print_active_op(log_dir=log_dir)

    def get_param_num_list(self):
        # recursively getting params
        return self.active_op.get_param_num_list()

    def get_flops_num_list(self, x):
        flops, x = self.active_op.get_flops_num_list()
        x = self.forward(x)

        return flops, x


class MnasNetSearchSpace(NASModule):
    """MnasNet search space is composed of first_conv, blocks, feature_mix_layer and classifier.

    There are no categorical variables belonging to this class.

    Args:
        name (str): Class name.
        is_search (bool, optional): Defaults to False.
    """

    def __init__(self, name: str, is_search: bool = False) -> None:
        super(__class__, self).__init__(name)

        self.n_classes = -1
        self.block_num = -1
        self.first_conv_filter_size = -1
        self.is_search = is_search

        # ModuleList of MnasNetBlock
        self.blocks = nn.ModuleList([])

        self.conv_head_candidate_dict = nn.ModuleDict({})

        self.feature_mix_layer = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.2)
        # candidates of classifier (fc layers that have different in_channels)

    def build(self, config: dict[str, Any], **kwargs) -> None:
        """Create supernet based on config file.

        Args:
            config (dict[str, Any]): A dict type config that can be created using util function.
        """

        self.n_classes = config["class_num"]
        self.block_num = config["block_num"]
        self.first_conv_filter_size = config["first_conv_filter_size"]
        self.first_mbconv_filter_size = config["first_mbconv_filter_size"]
        self.last_ch = config["last_ch"]

        # first conv
        self.first_conv = nn.Conv2d(3, self.first_conv_filter_size, 3, padding=1, stride=2, bias=False)
        self.first_batch = BatchNorm(self.first_conv_filter_size)
        self.first_relu = ReLUOp()

        self.first_mbconv = nn.Sequential(
            Conv(self.first_conv_filter_size, self.first_conv_filter_size, 3, 1, groups=self.first_conv_filter_size),
            Conv(self.first_conv_filter_size, self.first_mbconv_filter_size, k_size=1, stride=1, bn=True, relu=False),
        )
        # blocks

        for block_idx in range(self.block_num):
            name = f"Block{block_idx}"
            block = MnasNetBlock(name)
            block.build(config)
            self.blocks.append(block)

        # candidates of classifier
        classifier_in_chs = config["classifier_in_chs"]

        for in_ch in classifier_in_chs:
            self.conv_head_candidate_dict[str(in_ch)] = nn.Sequential(
                nn.Conv2d(in_ch, self.last_ch, 1, padding=0, stride=1, bias=False),
                BatchNorm(self.last_ch),
                ReLUOp(),
            )

        # With 200 classes when searching structure
        if self.is_search:
            self.classifier = nn.Linear(self.last_ch, 200, bias=True)
        else:
            self.classifier = nn.Linear(self.last_ch, self.n_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward function

        Args:
            x (torch.Tensor): An input tensor (batch size, input channel, height, width).

        Returns
            torch.Tensor: A tensor after operation (batch size, output channel, height, width).
        """

        x = self.first_conv(x)
        x = self.first_batch(x)
        x = self.first_relu(x)
        x = self.first_mbconv(x)
        for block in self.blocks:
            x = block(x)

        x = self.conv_head(x)
        x = self.feature_mix_layer(x)
        x = x.view(-1, self.last_ch)
        x = self.dropout(x)
        return self.classifier(x)

    def select_active_op(self, structure_info: MnasNetStructureInfo, **kwargs) -> None:
        """Set subnet based on structure_info. (There are no categorical variables belonging to this class.)

        Args:
            structure_info (MnasNetStructureInfo): Categorical keys/variables.
        """

        prev_filter_size = self.first_mbconv_filter_size
        # recursively setting
        for block in self.blocks:
            block.select_active_op(structure_info, prev_filter_size)
            prev_filter_size = block.filter_size

        # set classifier based on the filter_size of last block
        self.conv_head_in_ch = self.blocks[-1].filter_size
        self.conv_head = self.conv_head_candidate_dict[str(self.conv_head_in_ch)]

    def enumerate_categorical_variables(self) -> dict[str, Any]:
        """Create categorical variable dictionary.

        Returns:
            dict[str, Any]: Categorical variable dictionary
        """

        # This class does not have any categorical variables.
        categorical_variables = {}  # Python3.7以降は、辞書は追加順序保持

        # TODO: 次の子オブジェクトの辞書を回収するコードは重複するので、共通化を検討
        for block in self.blocks:
            sub_cat_vars = block.enumerate_categorical_variables()
            for key, var in sub_cat_vars.items():
                if key not in categorical_variables:
                    categorical_variables[key] = var

        return categorical_variables

    def enumerate_active_variables(self) -> dict[str, Any]:
        """Create active variable dictionary for debug.

        Returns:
            dict[str, Any]: Categorical variable dictionary.
        """

        # This class does not have any categorical variables.
        active_variables = {}  # Python3.7以降は、辞書は追加順序保持

        for block in self.blocks:
            sub_cat_vars = block.enumerate_active_variables()
            for key, var in sub_cat_vars.items():
                if key not in active_variables:
                    active_variables[key] = var

        return active_variables

    def fix_arc(self, **kwargs) -> None:
        """Deletes supernet instance."""
        self.conv_head_candidate_dict = nn.ModuleDict({})

        for block in self.blocks:
            block.fix_arc()

    def print_active_op(self, log_dir: Path | None = None):
        """Prints subnet information."""

        # first conv layer
        params = 3 * self.first_conv_filter_size * 3 * 3 + self.first_conv_filter_size * 2
        params_str = f"params: {params:>10} "
        kernel_size_str = "kernel_size: 3 "
        filter_size_str = f"filter_size: {self.first_conv_filter_size} "
        # print("[first conv]    " + params_str + kernel_size_str + filter_size_str)

        if log_dir is not None:
            with Path.open(log_dir / "architecture.txt", "a") as o:
                o.write("[first conv]    " + params_str + kernel_size_str + filter_size_str + "\n")

        # first MBconv layer
        params = self.first_mbconv[0].get_param_nums(
            self.first_conv_filter_size,
            self.first_conv_filter_size,
        ) + self.first_mbconv[1].get_param_nums(self.first_conv_filter_size, self.first_mbconv_filter_size)
        params_str = f"params: {params:>10} "
        kernel_size_str = "kernel_size: 1 "
        filter_size_str = f"filter_size: {self.first_mbconv_filter_size} "
        # print("[first MBconv]    " + params_str + kernel_size_str + filter_size_str)

        if log_dir is not None:
            with Path.open(log_dir / "architecture.txt", "a") as o:
                o.write("[first MBconv]    " + params_str + kernel_size_str + filter_size_str + "\n")

        # blocks
        for block in self.blocks:
            block.print_active_op(log_dir=log_dir)

        # conv_head+classifier
        params = (
            self.conv_head_in_ch * self.last_ch * 1 * 1
            + self.last_ch * 2
            + self.last_ch * self.n_classes
            + self.n_classes
        )
        params_str = f"params: {params:>10} "
        n_classes_str = f"class_num: {self.n_classes} "
        # print("[classifier]    " + params_str + n_classes_str)

        if log_dir is not None:
            with Path.open(log_dir / "architecture.txt", "a") as o:
                o.write("[classifier]    " + params_str + n_classes_str + "\n")

        # total params
        params_list = self.get_param_num_list()
        total_params = sum(sum(params) for params in params_list)

        # print("Total params: {}".format(total_params))
        if log_dir is not None:
            with Path.open(log_dir / "architecture.txt", "a") as o:
                o.write(f"Total params: {total_params}" + "\n")

    def get_param_num_list(self) -> list[Any]:
        """Gets a list of the number of parameters.

        Returns:
            list[Any]: [[first_conv params], [first_mb_conv params], [Block0 param list], [Block1 param list],
        ..., [classifier params]]
        """
        params_list = []

        # first conv layer
        first_mb_params = self.first_mbconv[0].get_param_nums(
            self.first_conv_filter_size,
            self.first_conv_filter_size,
        ) + self.first_mbconv[1].get_param_nums(self.first_conv_filter_size, self.first_mbconv_filter_size)
        params = [3 * self.first_conv_filter_size * 3 * 3 + self.first_conv_filter_size * 2, first_mb_params]
        params_list.append(params)

        # blocks
        for block in self.blocks:
            params = block.get_param_num_list()
            params_list.append(params)

        # conv_head+classifier
        params = [
            self.conv_head_in_ch * self.last_ch * 1 * 1
            + self.last_ch * 2
            + self.last_ch * self.n_classes
            + self.n_classes,
        ]
        params_list.append(params)

        return params_list

    def get_flops(self, size):
        x = torch.tensor(np.zeros(size)).cuda()
        flops_list = []

        x = self.first_conv(x)
        for block in self.blocks:
            flops, x = block.get_flops_list(x)
            flops_list.append(flops)
            x = self.forward(x)

        return flops_list
