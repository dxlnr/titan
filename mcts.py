import chess

from titan.mcts.node import Node

board = chess.Board()

lm = list(board.legal_moves)
print(len(lm), lm)

class State():
    """Defines the state of the specific game. In this case it is chess."""

    def __init__(self):
        pass

    def get_possible_moves(self):
        pass


state = board
s0 = Node(state)

def run_mcts(node: Node, state: chess.Board(), n_rollouts: int = 10):
    """Runs the Monte-Carlo Tree Search."""
    for i in range(n_rollouts):
        for a in state.legal_moves:
            node.expand(a)


run_mcts(s0, state)
