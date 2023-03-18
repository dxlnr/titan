import chess
import copy

from titan.mcts.node import Node
from titan.mcts.state import State

board = chess.Board()
state = State(board)

def run_mcts(state: State, n_rollouts: int = 100):
    """Runs the Monte-Carlo Tree Search."""
    root_node = Node()
    for i in range(n_rollouts):
        node, s = root_node, copy.deepcopy(state)
        
        # (1) Select
        while not node.is_leaf_node():
            node = policy(node)
            s.update(node.move)
        
        # (2) Expand
        node.expand(s)
        node = policy(node)

        # (3) Simulate
        while not s.is_terminated():
            s = simulate_policy(node)

        delta = s.eval()

        # (4) Backpropage
        while not n.is_root():
            n.update(delta)
            n = n.parent

        break 

run_mcts(state)
