"""Network Implementations"""
import torch
import torch.nn as nn

from titan.models.layers import Conv, Residual


class PolicyNet(nn.Module):
    def __init__(self, block, layers, **kwargs):
        super(PolicyNet, self).__init__()

        self.head = self._make_head()

        # Initialize the weights.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_head(self):
        return nn.Sequential(
            nn.Conv2d(
                in_c,
                out_c,
                kernel,
                stride,
                autopad(kernel, pad),
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Linear(512 * block.expansion, 19 * 19),
        )

    def forward(self):
        self.head(x)


def build_policy_network(block, layers, pretrained: bool = False, **kwargs):
    """."""
    model = PolicyNet(block, layers, **kwargs)

    if pretrained:
        model.load_state_dict(state_dict)
    return model


class ValueNet(nn.Module):
    def __init__(self):
        pass

    def forward(self):
        pass
