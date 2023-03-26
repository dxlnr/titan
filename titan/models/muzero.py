"""MuZero Network"""
import torch
import torch.nn as nn

from titan.config import Conf
from titan.mcts.state import State
from titan.models.nets import ReprNet


class M0Net(nn.Module):
    """MuZero Network Architecture"""

    def __init__(self, cfg: Conf):
        super(M0Net, self).__init__()
        self.cfg = cfg

        # Representation function that encodes past observations.
        self.repr_network = ReprNet(
            self.cfg.OBSERVATION_SHAPE[2], self.cfg.CHANNELS, self.cfg.DEPTH
        )
        #
        self.dyn_network = None
        #
        self.prediction_network = None

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
        # Stack encoded_state with a game specific one hot encoded action.
        # (See paper appendix Network Architecture)
        pass

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
        policy_logits, value = self.prediction(next_s)

        return value, reward, policy_logits, next_s
