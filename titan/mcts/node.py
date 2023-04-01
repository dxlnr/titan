"""Node Object for MCTS"""
import math
import numpy as np
import torch

from titan.mcts.state import State


class Node:
    """Node within the MCTS."""

    def __init__(self, prior: float, action: str = "", parent=None):
        self.n_k, self.w_k = 0, 0
        self.reward = 0
        self.to_play = -1
        self.action = action
        self.parent = parent
        self.prior = prior
        self.children = list()

    def __repr__(self) -> str:
        return f"Node| n: {self.n_k}, w: {self.w_k}, action: {self.action}, parent: {self.parent}."

    def expand(self, actions: int, to_play, reward, policy_logits, hidden_state) -> None:
        """When node is choosen and not terminal, expands the tree by finding all
        possible child nodes.

        :param state: State attached to this specific node.
        """
        self.to_play = to_play
        self.reward = reward
        self.hidden_state = hidden_state

        print(policy_logits)
        print(policy_logits.shape)

        print(type(actions))
        print(actions)
        # if not state.is_terminal():
        policy_values = torch.softmax(policy_logits, dim=0).tolist()

        print("policy values: ", policy_values)
        policy = {a: policy_values[i] for i, a in enumerate(actions)}

        for action, p in policy.items():
            self.children.append(Node(p, action, self))
            # for m in state.get_legal_actions():
            #     c_node = Node(m, self)
            #     self.children.append(c_node)

    def add_exploration_noise(self, root_dirichlet_alpha, root_exploration_frac):
        """At the start of each search, dirichlet noise is added to the prior of
        the root to encourage the search to explore new actions.

        :param root_dirichlet_alpha:
        :param root_exploration_frac:
        """
        noise = np.random.dirichlet([root_dirichlet_alpha] * len(self.children))
        for n in noise:
            self.children.prior = (
                node.children.prior * (1 - root_exploration_frac)
                + n * root_exploration_frac
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
