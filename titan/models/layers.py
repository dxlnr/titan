"""Residual Connection"""
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_divisible(
    v, divisor: int = 8, min_value: float = None, round_limit: float = 0.9
):
    """."""
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    if new_v < round_limit * v:
        new_v += divisor
    return new_v


def get_padding(kernel: int, stride: int = 1, dilation: int = 1) -> int:
    """Calculate symmetric padding for a convolution."""
    pad = ((stride - 1) + dilation * (kernel - 1)) // 2
    return pad


class StdConv2d(nn.Conv2d):
    """Conv2d with weight Standardization.

    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight Standardization` -
        https://arxiv.org/abs/1903.10520v2
    """

    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel: int = 3,
        stride: int = 1,
        pad: int = None,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        eps: float = 1e-6,
    ):
        if pad is None:
            pad = get_padding(kernel, stride, dilation)
        super().__init__(
            in_c,
            out_c,
            kernel_size=kernel,
            stride=stride,
            padding=pad,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.eps = eps

    def forward(self, x):
        weight = F.batch_norm(
            self.weight.reshape(1, self.out_c, -1),
            None,
            None,
            training=True,
            momentum=0.0,
            eps=self.eps,
        ).reshape_as(self.weight)
        x = F.conv2d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.

    Follows the implementation of "Identity Mappings in Deep Residual Networks":
    https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua

    Except it puts the stride on 3x3 conv when available.
    """

    def __init__(
        self,
        in_c,
        out_c=None,
        bottle_ratio: float = 0.25,
        stride: int = 1,
        dilation: int = 1,
        first_dilation: int = None,
        groups: int = 1,
        act_layer=None,
        conv_layer=None,
        norm_layer=None,
        proj_layer=None,
        drop_path_rate=0.0,
    ):
        super().__init__()

        first_dilation = first_dilation or dilation
        conv_layer = conv_layer or StdConv2d
        norm_layer = norm_layer or partial(GroupNormAct, num_groups=32)

        out_c = out_c or in_c
        mid_c = make_divisible(out_c * bottle_ratio)

        if proj_layer is not None:
            self.downsample = proj_layer(
                in_c,
                out_c,
                stride=stride,
                dilation=dilation,
                first_dilation=first_dilation,
                preact=True,
                conv_layer=conv_layer,
                norm_layer=norm_layer,
            )
        else:
            self.downsample = None

        self.norm1 = norm_layer(in_c)
        self.conv1 = conv_layer(in_c, mid_c, 1)
        self.norm2 = norm_layer(mid_c)
        self.conv2 = conv_layer(
            mid_c, mid_c, 3, stride=stride, dilation=first_dilation, groups=groups
        )
        self.norm3 = norm_layer(mid_c)
        self.conv3 = conv_layer(mid_c, out_c, 1)
        # self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def zero_init_last(self):
        nn.init.zeros_(self.conv3.weight)

    def forward(self, x):
        x_preact = self.norm1(x)

        # skip connection
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x_preact)

        # residual branch
        x = self.conv1(x_preact)
        x = self.conv2(self.norm2(x))
        x = self.conv3(self.norm3(x))
        # x = self.drop_path(x)
        return x + identity


class Residual(nn.Module):
    """."""

    def __init__(
        self,
        in_c,
        out_c=None,
        stride: int = 1,
        dilation: int = 1,
        first_dilation: int = None,
        groups: int = 1,
        act_layer=None,
        conv_layer=None,
        norm_layer=None,
        proj_layer=None,
        drop_path_rate=0.0,
    ):
        super().__init__()

        first_dilation = first_dilation or dilation
        conv_layer = conv_layer or StdConv2d
        norm_layer = norm_layer or nn.BatchNorm2d

        out_c = out_c or in_c

        if proj_layer is not None:
            self.downsample = proj_layer(
                in_c,
                out_c,
                stride=stride,
                dilation=dilation,
                first_dilation=first_dilation,
                preact=True,
                conv_layer=conv_layer,
                norm_layer=norm_layer,
            )
        else:
            self.downsample = None

        self.norm1 = norm_layer(in_c)
        self.conv1 = conv_layer(in_c, out_c, 1)
        self.norm2 = norm_layer(out_c)
        self.conv2 = conv_layer(
            out_c, out_c, 3, stride=stride, dilation=first_dilation, groups=groups
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # skip connection
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        # residual branch
        x = self.conv1(x)
        x = self.conv2(self.norm2(x))
        return self.relu(x + identity)


class DownsampleAvg(nn.Module):
    """Average Downsampling Layer."""

    def __init__(
        self,
        in_c,
        out_c,
        stride=1,
        dilation=1,
        first_dilation=None,
        preact=True,
        conv_layer=None,
        norm_layer=None,
    ):
        """AvgPool Downsampling as in 'D' ResNet variants."""
        super(DownsampleAvg, self).__init__()

        avg_stride = stride if dilation == 1 else 1
        if stride > 1 or dilation > 1:
            avg_pool_fn = (
                AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
            )
            self.pool = avg_pool_fn(
                2, avg_stride, ceil_mode=True, count_include_pad=False
            )
        else:
            self.pool = nn.Identity()
        self.conv = conv_layer(in_chs, out_chs, 1, stride=1)
        self.norm = nn.Identity() if preact else norm_layer(out_chs, apply_act=False)

    def forward(self, x):
        return self.norm(self.conv(self.pool(x)))


# class ResNetStage(nn.Module):
#     """ResNet Stage."""

#     def __init__(
#         self,
#         in_c,
#         out_c,
#         depth,
#         stride,
#         dilation,
#         bottle_ratio=0.25,
#         groups=1,
#         avg_down=False,
#         block_dpr=None,
#         block_fn=PreActBottleneck,
#         act_layer=None,
#         conv_layer=None,
#         norm_layer=None,
#         **block_kwargs,
#     ):
#         super(ResNetStage, self).__init__()

#         first_dilation = 1 if dilation in (1, 2) else 2
#         layer_kwargs = dict(
#             act_layer=act_layer, conv_layer=conv_layer, norm_layer=norm_layer
#         )
#         proj_layer = DownsampleAvg if avg_down else DownsampleConv
#         prev_c = in_c

#         self.blocks = nn.Sequential()
#         for idx in range(depth):
#             drop_path_rate = block_dpr[idx] if block_dpr else 0.0
#             stride = stride if idx == 0 else 1

#             self.blocks.add_module(
#                 str(idx),
#                 block_fn(
#                     prev_c,
#                     out_c,
#                     stride=stride,
#                     dilation=dilation,
#                     bottle_ratio=bottle_ratio,
#                     groups=groups,
#                     first_dilation=first_dilation,
#                     proj_layer=proj_layer,
#                     drop_path_rate=drop_path_rate,
#                     **layer_kwargs,
#                     **block_kwargs,
#                 ),
#             )
#             prev_c = out_c
#             first_dilation = dilation
#             proj_layer = None

#     def forward(self, x):
#         x = self.blocks(x)
#         return x
