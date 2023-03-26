"""Network Implementations"""
import torch
import torch.nn as nn

from titan.models.layers import StdConv2d, Residual


class ReprNet(nn.Module):
    """."""

    def __init__(
        self,
        c_in,
        c_out,
        depth=16,
        stride=1,
        dilation=1,
        groups=1,
        block_fn=Residual,
        act_layer=nn.ReLU,
        conv_layer=None,
        norm_layer=None,
        **block_kwargs
    ):
        super(ReprNet, self).__init__()

        layer_kwargs = dict(
            act_layer=act_layer, conv_layer=conv_layer, norm_layer=norm_layer
        )
        first_dilation = 1 if dilation in (1, 2) else 2

        self.act_layer = act_layer
        self.conv = StdConv2d(c_in, c_out)
        self.blocks = nn.Sequential()

        c_prev = c_in
        for idx in range(depth):
            # drop_path_rate = block_dpr[idx] if block_dpr else 0.0
            stride = stride if idx == 0 else 1

            self.blocks.add_module(
                str(idx),
                block_fn(
                    c_prev,
                    c_out,
                    stride=stride,
                    dilation=dilation,
                    first_dilation=first_dilation,
                    groups=groups,
                    **layer_kwargs,
                    **block_kwargs,
                ),
            )
            c_prev = c_out
            first_dilation = dilation

    def forward(self, x):
        x = self.act_layer(self.conv(x))
        x = self.blocks(x)
        return x


class PredictionNet(nn.Module):
    """."""

    def __init__(
        self,
        c_in,
        depth,
        reduced_c_value,
        reduced_c_policy,
        fc_value_layers,
        fc_policy_layers,
        full_support_size,
        block_output_value,
        block_output_policy,
        **kwargs
    ):
        super(PolicyNet, self).__init__()

        self.block_output_value = block_output_value
        self.block_output_policy = block_output_policy

        self._make_head()
        self.blocks = nn.Sequential()

        c_out = c_in
        for idx in range(depth):
            self.blocks.add_module(
                str(idx),
                block_fn(
                    c_in,
                    c_out,
                ),
            )

    def _make_head(self):
        self.conv_value = nn.Conv2d(c_in, reduced_c_value, 1)
        self.conv_policy = nn.Conv2d(c_in, reduced_c_policy, 1)

        self.fc_value = mlp(self.block_output_value, fc_value_layers, full_support_size)
        self.fc_policy = mlp(
            self.block_output_policy,
            fc_policy_layers,
            action_space_size,
        )

    def forward(self):
        x = self.block(x)

        value = self.conv_value(x)
        policy = self.conv_policy(x)

        value = value.view(-1, self.block_output_value)
        policy = policy.view(-1, self.block_output_policy)

        value = self.fc_value(value)
        policy = self.fc_policy(policy)
        return policy, value


# def build_prediction_network(pretrained: bool = False, **kwargs):
#     """."""
#     model = PredictionNet(block, layers, **kwargs)

#     if pretrained:
#         model.load_state_dict(state_dict)
#     return model


class DynamicsNet(nn.Module):
    def __init__(
        self,
        in_c,
        depth,
        reduced_c_rewards,
        fc_reward_layers,
        full_support_size,
        block_output_reward,
    ):
        super().__init__()
        # self.conv = conv3x3(num_channels, num_channels - 1)
        self.conv = StdConv2d(in_c, in_c - 1)
        self.bn = nn.BatchNorm2d(in_c - 1)

        self.blocks = nn.Sequential()
        for idx in range(depth):
            self.blocks.add_module(
                str(idx),
                block_fn(
                    c_in,
                    c_out,
                ),
            )

        self.conv_reward = nn.Conv2d(in_c - 1, reduced_c_reward, 1)
        self.block_output_reward = block_output_reward

        self.fc = mlp(
            self.block_output_reward,
            fc_reward_layers,
            full_support_size,
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = nn.functional.relu(x)

        x = self.block(x)
        state = x
        x = self.conv_reward(x)
        x = x.view(-1, self.block_output_reward)
        reward = self.fc(x)
        return state, reward
