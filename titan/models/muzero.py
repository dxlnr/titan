"""MuZero"""
import torch


class MuZero:
    def __init__(self):
        # Representation function that encodes past observations.
        self.repr = None
        #
        self.dynamics = None
        #
        self.prediction = None


def compute_repr():
    """Representation function that encodes past observations. Embeddings space
    without any particular semantics attached besides the support for future
    prediction."""
    pass


def compute_dynamics():
    """Dynamics function computes, at each step k, an immediate reward rk
    and an internal state sk."""
    pass


def compute_predicions():
    """Prediction function computes the policy and value function."""
    pass
