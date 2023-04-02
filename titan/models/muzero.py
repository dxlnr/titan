"""MuZero Network"""
import torch
import torch.nn as nn

from titan.config import Conf
from titan.mcts.state import State
from titan.models.nets import DynamicsNet, PredictionNet, ReprNet


class M0Net(nn.Module):
    """MuZero Network Architecture."""

    def __init__(self, cfg: Conf):
        super(M0Net, self).__init__()
        self.cfg = cfg

        downsample = False
        self.full_support_size = 2 * self.cfg.SUPPORT_SIZE + 1
        self.block_output_size_reward = (
            (
                self.cfg.REDUCED_C_REWARD
                * math.ceil(self.cfg.OBSERVATION_SHAPE[1] / 16)
                * math.ceil(self.cfg.OBSERVATION_SHAPE[2] / 16)
            )
            if downsample
            else (
                self.cfg.REDUCED_C_REWARD
                * self.cfg.OBSERVATION_SHAPE[1]
                * self.cfg.OBSERVATION_SHAPE[2]
            )
        )

        self.block_output_size_value = (
            (
                self.cfg.REDUCED_C_VALUE
                * math.ceil(self.cfg.OBSERVATION_SHAPE[1] / 16)
                * math.ceil(self.cfg.OBSERVATION_SHAPE[2] / 16)
            )
            if downsample
            else (
                self.cfg.REDUCED_C_VALUE
                * self.cfg.OBSERVATION_SHAPE[1]
                * self.cfg.OBSERVATION_SHAPE[2]
            )
        )

        self.block_output_size_policy = (
            (
                self.cfg.REDUCED_C_POLICY
                * math.ceil(self.cfg.OBSERVATION_SHAPE[1] / 16)
                * math.ceil(self.cfg.OBSERVATION_SHAPE[2] / 16)
            )
            if downsample
            else (
                self.cfg.REDUCED_C_POLICY
                * self.cfg.OBSERVATION_SHAPE[1]
                * self.cfg.OBSERVATION_SHAPE[2]
            )
        )
        # Representation function that encodes past observations.
        self.repr_network = ReprNet(
            self.cfg.OBSERVATION_SHAPE[0], self.cfg.CHANNELS, self.cfg.DEPTH
        )
        #
        self.dyn_network = DynamicsNet(
            (self.cfg.CHANNELS + self.cfg.ACTION_SHAPE[0]),
            self.cfg.CHANNELS,
            self.cfg.DEPTH,
            self.cfg.REDUCED_C_REWARD,
            self.cfg.RESNET_FC_REWARD_LAYERS,
            self.full_support_size,
            self.block_output_size_reward,
        )
        #
        self.prediction_network = PredictionNet(
            self.cfg.CHANNELS,
            self.cfg.DEPTH,
            self.cfg.ACTION_SPACE,
            self.cfg.REDUCED_C_VALUE,
            self.cfg.REDUCED_C_POLICY,
            self.cfg.RESNET_FC_VALUE_LAYERS,
            self.cfg.RESNET_FC_POLICY_LAYERS,
            self.full_support_size,
            self.block_output_size_value,
            self.block_output_size_policy,
        )

        self.full_support_size = 10

    def representation(self, obs):
        """."""
        s = self.repr_network(obs)

        # Scale encoded state between [0, 1] (See appendix paper Training)
        min_s = (
            s.view(-1, s.shape[1], s.shape[2] * s.shape[3])
            .min(2, keepdim=True)[0]
            .unsqueeze(-1)
        )
        max_s = (
            s.view(-1, s.shape[1], s.shape[2] * s.shape[3])
            .max(2, keepdim=True)[0]
            .unsqueeze(-1)
        )

        scale_s = max_s - min_s
        scale_s[scale_s < 1e-5] += 1e-5
        s_norm = (s - min_s) / scale_s
        return s_norm

    def prediction(self, hidden_state: torch.Tensor):
        """Prediction function computes the policy and value function."""
        policy, value = self.prediction_network(hidden_state)
        return policy, value

    def dynamics(self, hidden_state, action):
        """."""
        print(len(action.shape))
        if len(action.shape) == 3:
            action = action.unsqueeze(0)
        if len(hidden_state.shape) == 3:
            hidden_state = hidden_state.unsqueeze(0)

        print(hidden_state.shape, action.shape)
        # Stack encoded_state with a game specific one hot encoded action.
        # (See paper appendix Network Architecture)
        x = torch.cat((hidden_state, action), 1)

        s, reward = self.dyn_network(x)

        # Scale encoded state between [0, 1] (See appendix paper Training)
        min_s = (
            s.view(-1, s.shape[1], s.shape[2] * s.shape[3])
            .min(2, keepdim=True)[0]
            .unsqueeze(-1)
        )
        max_s = (
            s.view(-1, s.shape[1], s.shape[2] * s.shape[3])
            .max(2, keepdim=True)[0]
            .unsqueeze(-1)
        )
        scale_s = max_s - min_s
        scale_s[scale_s < 1e-5] += 1e-5
        next_s_norm = (s - min_s) / scale_s

        return next_s_norm, reward

    def initial_inference(self, obs):
        s = self.representation(obs)
        policy_logits, value = self.prediction(s)

        # reward equal to 0 for consistency
        reward = torch.log(
            (
                torch.zeros(1, self.full_support_size)
                .scatter(1, torch.tensor([[self.full_support_size // 2]]).long(), 1.0)
                .repeat(len(obs), 1)
                .to(obs.device)
            )
        )
        return (
            value,
            reward,
            policy_logits,
            s,
        )

    def recurrent_inference(self, hidden_state, action):
        """."""
        next_s, reward = self.dynamics(hidden_state, action)
        print("next state shape: ", next_s.shape)
        policy_logits, value = self.prediction(next_s)

        return value, reward, policy_logits, next_s
