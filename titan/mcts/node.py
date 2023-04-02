"""Node Object for MCTS"""
import math
import numpy as np
import torch

from titan.mcts.state import State


class Node:
    """Node within the MCTS."""

    def __init__(self, prior: float, parent=None):
        self.n_k, self.w_k = 0, 0
        self.reward = 0
        self.to_play = -1
        self.hidden_state = None
        self.parent = parent
        self.prior = prior
        self.children = {}

    def __repr__(self) -> str:
        return f"Node| n: {self.n_k}, w: {self.w_k}, action: {self.action}, parent: {self.parent}."

    def expand(
        self, actions, to_play, reward, policy_logits, hidden_state
    ) -> None:
        """When node is choosen and not terminal, expands the tree by finding all
        possible child nodes.

        :param state: State attached to this specific node.
        """
        self.to_play = to_play
        self.reward = reward
        self.hidden_state = hidden_state

        if len(policy_logits.shape) == 2:
            policy_logits = policy_logits.squeeze()

        policy = {a: math.exp(policy_logits[a]) for a in actions}
        policy_sum = sum(policy.values())

        for action, p in policy.items():
            self.children[action] = Node((p / policy_sum), self)

    def add_exploration_noise(self, config):
        """At the start of each search, dirichlet noise is added to the prior of
        the root to encourage the search to explore new actions.

        :param root_dirichlet_alpha:
        :param root_exploration_frac:
        """
        actions = list(self.children.keys())
        noise = np.random.dirichlet([config.ROOT_DIRICHLET_ALPHA] * len(self.children))
        for a, n in zip(actions, noise):
            self.children[a].prior = (
                self.children[a].prior * (1 - config.ROOT_EXPLORATION_FRACTION)
                + n * config.ROOT_EXPLORATION_FRACTION
            )

    def is_leaf_node(self) -> bool:
        """Returns True if the node is a leaf node."""
        return len(self.children) == 0

    def is_root_node(self) -> bool:
        """Returns True is the node is a root node."""
        return self.parent is None

    def propagate(self, r: int) -> None:
        """Propagate back through the tree and update internal values accordingly.

        :param r: Reward for based on the outcome following down this path.
        """
        self.n_k = self.n_k + 1
        self.w_k = self.w_k + r

    def uct(self) -> float:
        """Computes the UCT (Upper Confidence Bound 1 applied to trees) value."""
        return (self.w_k / self.n_k) + 2 * math.sqrt(
            (math.log(self.parent.n_k) / self.n_k)
        )

    def value(self) -> float:
        """Returns the node value based on the fraction of reward and visits."""
        if self.n_k == 0:
            return 0
        return self.w_k / self.n_k
