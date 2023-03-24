"""Residual Connection"""
import torch
import torch.nn as nn


class Conv(nn.Module):
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


class Residual(nn.Module):
    def __init__(
        self, in_c: int, out_c: int, kernel: int = 3, stride: int = 1, pad: int = None
    ):
        super().__init__()
        self.c1 = Conv(in_c, out_c, kernel, stride, pad)
        self.c2 = Conv(in_c, out_c, kernel, stride, pad, act=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.c2(self.c1(x))
        out += x
        return self.relu(out)
