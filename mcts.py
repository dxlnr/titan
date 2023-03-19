"""Monte-Carlo Tree Search"""
import chess
import copy
import random 

from titan.mcts.node import Node
from titan.mcts.state import State

board = chess.Board()
state = State(board)


def policy(node: Node) -> Node:
    """Policy that defines which node gets choosen next.

    The formula which is used is called UCT (Upper Confidence Bound 1 applied to trees).
    The node with the highest UCT gets choosen. 
    """
    idx = 0
    ucb = 0
    for i, n in enumerate(node.children):
        if n.n_k == 0:
            idx = i
            break
        else:
            if (val := n.ucb()) > ucb:
                idx = i

    return node.children[idx]


def simulate_policy(state: State) -> State:
    """."""
    state.random_upd()
    return state


def run_mcts(state: State, n_rollouts: int = 500):
    """Runs the Monte-Carlo Tree Search."""
    root_node = Node()
    for i in range(n_rollouts):
        node, s = root_node, copy.deepcopy(state)

        # (1) Select
        while not node.is_leaf_node():
            node = policy(node)
            s.update(str(node.move))

        # (2) Expand
        node.expand(s)
        node = policy(node)

        # (3) Simulate
        while not s.is_terminated():
            s = simulate_policy(s)
        delta = s.eval()

        # (4) Backpropagate
        while True:
            node.propagate(delta)
            if node.is_root_node():
                break
        
            node = node.parent


run_mcts(state)
