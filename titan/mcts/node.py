"""Node Object for MCTS"""
import math
from titan.mcts.state import State


class Node:
    def __init__(self, move: str = "", parent=None):
        self.n_k, self.w_k = 0, 0
        self.move = move
        self.parent = parent
        self.children = list()

    def __repr__(self) -> str:
        return f"Node| n: {self.n_k}, w: {self.w_k}, move: {self.move}, parent: {self.parent}."

    def expand(self, state: State) -> None:
        """When node is choosen and not terminal, expands the tree by finding all
        possible child nodes.

        :param state: State attached to this specific node.
        """
        if not state.is_terminal():
            for m in state.get_legal_actions():
                c_node = Node(m, self)
                self.children.append(c_node)

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """."""
        pass

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

    def value(self):
        pass
