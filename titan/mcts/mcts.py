"""Monte-Carlo Tree Search"""
import copy
import random
from tqdm import tqdm

from titan.mcts.state import State
from titan.mcts.node import Node


def policy(node: Node) -> Node:
    """Policy that defines which node gets choosen next.

    The formula which is used is called UCT (Upper Confidence Bound 1 applied to trees).
    The node with the highest UCT gets choosen.
    """
    idx = 0
    uct = 0
    for i, n in enumerate(node.children):
        if n.n_k == 0:
            idx = i
            break
        else:
            if (val := n.uct()) > uct:
                idx = i
                uct = val

    return node.children[idx]


def simulate_policy(state: State) -> State:
    """."""
    state.random_upd()
    return state


def choose_move(node: Node, flag: str = "") -> str:
    """."""

    def score(n):
        return n.w_k / n.n_k

    if flag == "v":
        from operator import attrgetter

        cn = max(node.children, key=attrgetter("n_k"))
    else:
        cn = max(node.children, key=score)

    return cn.move


def mcts(state: State, n_rollouts: int = 250):
    """Runs the Monte-Carlo Tree Search."""
    root_node = Node()
    for i in tqdm(range(n_rollouts)):
        node, s = root_node, copy.deepcopy(state)

        # (1) Select
        while not node.is_leaf_node():
            node = policy(node)
            s.update(str(node.move))

        # (2) Expand
        node.expand(s)
        if not node.is_leaf_node() or not s.is_terminal():
            node = policy(node)

        # (3) Simulate
        while not s.is_terminal():
            s = simulate_policy(s)
        delta = s.eval()

        # (4) Backpropagate
        while True:
            node.propagate(delta)
            if node.is_root_node():
                break

            node = node.parent

    return choose_move(root_node)
