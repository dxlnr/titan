"""MuZero"""
import torch
import torch.nn as nn

from titan.mcts.state import State


class MuZero:
    def __init__(self):
        # Representation function that encodes past observations.
        self.repr = None
        #
        self.dynamics = None
        #
        self.prediction = None


class Representation(nn.Module):
    def __init__(self):
        self.emb = nn.Embeddings()

    def forward(self, s):
        pass


def compute_repr(state: State):
    """Representation function that encodes past observations.

    Embeddings space without any particular semantics attached
    besides the support for future prediction.
    """
    h = state

    return h


def compute_dynamics():
    """Dynamics function computes, at each step k, an immediate reward rk
    and an internal state sk."""
    pass


def compute_predicions():
    """Prediction function computes the policy and value function."""
    pass
