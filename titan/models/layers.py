"""Layer Blocks for Networks"""
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


def autopad(kernel, pad=None, dilation=1):
    """Pad to 'same' shape outputs."""
    if dilation > 1:
        kernel = (
            dilation * (kernel - 1) + 1
            if isinstance(kernel, int)
            else [dilation * (x - 1) + 1 for x in kernel]
        )
    if pad is None:
        pad = kernel // 2 if isinstance(kernel, int) else [x // 2 for x in kernel]
    return pad


def get_padding(kernel: int, stride: int = 1, dilation: int = 1) -> int:
    """Calculate symmetric padding for a convolution."""
    pad = ((stride - 1) + dilation * (kernel - 1)) // 2
    return pad


def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=torch.nn.Identity,
    activation=torch.nn.ELU,
):
    """Basic Multi-Layer Perceptron Architecture."""
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
    return torch.nn.Sequential(*layers)


class Conv(nn.Module):
    """Convolution Block."""

    def __init__(
        self,
        in_c: int,
        out_c: int,
        kernel: int = 3,
        stride: int = 1,
        pad: int = None,
        groups: int = 1,
        act: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_c, out_c, kernel, stride, autopad(kernel, pad), groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = (
            nn.ReLU()
            if act is True
            else (act if isinstance(act, nn.Module) else nn.Identity())
        )

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


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
        self.out_c = out_c
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


class ResidualBlock(nn.Module):
    """Residual Block implementation."""

    def __init__(self, in_c, out_c, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = Conv(in_c, out_c)
        self.conv2 = Conv(out_c, out_c, act=False)
        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample:
            residual = self.downsample(x)
        out += identity
        return self.relu(out)
