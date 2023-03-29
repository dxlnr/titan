"""Monte-Carlo Tree Search"""
import copy
import random
from tqdm import tqdm

from titan.config import Conf
from titan.mcts.state import State
from titan.mcts.node import Node
from titan.mcts.action import ActionState
from titan.models import M0Net


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


def select_action(node: Node, temperature: float = 0) -> str:
    """."""

    def score(n):
        return n.w_k / n.n_k

    if temperature == 0:
        from operator import attrgetter

        cn = max(node.children, key=attrgetter("n_k"))
    else:
        cn = max(node.children, key=score)

    return cn.move


def run_mcts(
    config: Conf,
    root_node: Node,
    action_state: ActionState,
    model: M0Net,
    add_exploration_noise=False,
):
    """Runs the Monte-Carlo Tree Search."""
    # root_node = Node()

    # if add_exploration_noise:
    #     root_node.add_exploration_noise(
    #         dirichlet_alpha=config.ROOT_DIRICHLET_ALPHA,
    #         exploration_fraction=config.ROOT_EXPLORATION_FRACTION,
    #     )

    for i in tqdm(range(config.NUM_ROLLOUTS)):
        node, s = root_node, copy.deepcopy(action_state)
        search_path = [node]

        # (1) Select
        while not node.is_leaf_node():
            node = policy(node)
            s.update(str(node.action))
            # search_path.append(node)

        parent = search_path[-2]
        v, r, p, s_next = model.recurrent_inference(parent.s, game.last_action())

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

    # return choose_move(root_node)
    return root_node
