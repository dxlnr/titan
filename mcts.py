import chess
import copy

from titan.mcts.node import Node
from titan.mcts.state import State

board = chess.Board()
state = State(board)


def policy(node: Node) -> Node:
    """Policy network that defines which node gets choosen next."""
    print(len(node.children))
    idx = node.children.index(
        max(node.children, key=lambda x: x.ucb() if x.n_k != 0 else float("inf"))
    )
    
    # idx = 
    # for i, n in enumerate(node.children):
    #     if n.n_k == 0:
            

    print(idx)
    return node.children[idx]


def simulate_policy(node: Node) -> State:
    pass

def run_mcts(state: State, n_rollouts: int = 5):
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
        print(node)

        # (3) Simulate
        # while not s.is_terminated():
        #     s = simulate_policy(node)

        # delta = s.eval()

        delta = 1
        # (4) Backpropage
        while not node.is_root_node():
            node.propagate(delta)
            node = node.parent

run_mcts(state)
