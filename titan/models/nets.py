"""Network Implementations"""
import torch
import torch.nn as nn

from titan.models.layers import mlp, Conv, StdConv2d, ResidualBlock


def conv3x3(in_channels, out_channels, stride=1):
    return torch.nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


class ReprNet(nn.Module):
    """."""

    def __init__(
        self,
        c_in,
        c_out,
        depth,
        # stride=1,
        # dilation=1,
        # groups=1,
        # block_fn=ResidualBlock,
        # act_layer=nn.ReLU,
        # conv_layer=None,
        # norm_layer=None,
        # **block_kwargs
    ):
        super(ReprNet, self).__init__()

        # layer_kwargs = dict(
        #     act_layer=act_layer, conv_layer=conv_layer, norm_layer=norm_layer
        # )
        # first_dilation = 1 if dilation in (1, 2) else 2

        # self.act_layer = act_layer
        # self.conv = conv3x3(c_in, c_out)
        self.conv = Conv(c_in, c_out, stride=1)
        # self.bn = nn.BatchNorm2d(c_out)
        self.act_layer = nn.ReLU
        self.blocks = nn.Sequential()

        # self.resblocks = torch.nn.ModuleList(
        #     [ResidualBlock(c_out, c_out) for _ in range(depth)]
        # )

        for idx in range(depth):
            self.blocks.add_module(str(idx), ResidualBlock(c_out, c_out))
        # c_prev = c_in
        # for idx in range(depth):
        #     # drop_path_rate = block_dpr[idx] if block_dpr else 0.0
        #     stride = stride if idx == 0 else 1

        #     self.blocks.add_module(
        #         str(idx),
        #         block_fn(
        #             c_prev,
        #             c_out,
        #             stride=stride,
        #             # dilation=dilation,
        #             # first_dilation=first_dilation,
        #             # groups=groups,
        #             # **layer_kwargs,
        #             # **block_kwargs,
        #         ),
        #     )
        #     c_prev = c_out
        #     first_dilation = dilation

    def forward(self, x):
        # x = self.act_layer(self.conv(x))
        x = self.conv(x)

        # x = self.bn(x)
        # x = self.relu(x)
        x = self.blocks(x)
        return x


class PredictionNet(nn.Module):
    """."""

    def __init__(
        self,
        c_in,
        depth,
        action_space,
        reduced_c_value,
        reduced_c_policy,
        fc_value_layers,
        fc_policy_layers,
        full_support_size,
        block_output_value,
        block_output_policy,
        block_fn=ResidualBlock,
        **kwargs
    ):
        super(PredictionNet, self).__init__()

        # Network Parameter
        self.c_in = c_in
        self.action_space = action_space
        self.reduced_c_value = reduced_c_value
        self.reduced_c_policy = reduced_c_policy
        self.fc_value_layers = list(fc_value_layers)
        self.fc_policy_layers = list(fc_policy_layers)
        self.full_support_size = full_support_size
        self.block_output_value = block_output_value
        self.block_output_policy = block_output_policy

        # Prediction Head
        self._make_head()
        # Residual Blocks
        self.blocks = nn.Sequential()
        for idx in range(depth):
            self.blocks.add_module(
                str(idx),
                block_fn(
                    c_in,
                    c_in,
                ),
            )

    def _make_head(self):
        """."""
        self.conv_value = nn.Conv2d(self.c_in, self.reduced_c_value, 1)
        self.conv_policy = nn.Conv2d(self.c_in, self.reduced_c_policy, 1)

        self.fc_value = mlp(
            self.block_output_value, self.fc_value_layers, self.full_support_size
        )
        self.fc_policy = mlp(
            self.block_output_policy,
            self.fc_policy_layers,
            self.action_space,
        )

    def forward(self, x):
        x = self.blocks(x)
        value = self.conv_value(x)
        policy = self.conv_policy(x)
        value = value.view(-1, self.block_output_value)
        policy = policy.view(-1, self.block_output_policy)
        value = self.fc_value(value)
        policy = self.fc_policy(policy)
        return policy, value


class DynamicsNet(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        depth,
        reduced_c_rewards,
        fc_reward_layers,
        full_support_size,
        block_output_reward,
        block_fn=ResidualBlock,
    ):
        super(DynamicsNet, self).__init__()
        self.fc_reward_layers = list(fc_reward_layers)

        self.conv = Conv(c_in, c_out)
        self.blocks = nn.Sequential()
        for idx in range(depth):
            self.blocks.add_module(
                str(idx),
                block_fn(
                    c_out,
                    c_out,
                ),
            )

        self.conv_reward = nn.Conv2d(c_out, reduced_c_rewards, 1)
        self.block_output_reward = block_output_reward

        self.fc = mlp(
            self.block_output_reward,
            self.fc_reward_layers,
            full_support_size,
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.blocks(x)
        state = x
        x = self.conv_reward(x)
        x = x.view(-1, self.block_output_reward)
        reward = self.fc(x)
        return state, reward
