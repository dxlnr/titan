"""Monte-Carlo Tree Search"""
import copy
import numpy as np
import random
import math
from tqdm import tqdm

from titan.config import Conf
from titan.mcts.state import State
from titan.mcts.node import Node
from titan.mcts.action import ActionHistory
from titan.models.muzero import transform_to_scalar

# from titan.game.chess_state import Chess
from titan.models import M0Net


class MinMaxStats:
    """A class that holds the min-max values of the tree."""

    MAXIMUM_FLOAT_VALUE: float = float("inf")

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


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


def select_child(config, node: Node, min_max_stats: MinMaxStats):
    """Select the child with the highest UCB score."""
    max_ucb = max(
        ucb_score(config, node, child, min_max_stats)
        for action, child in node.children.items()
    )
    action = np.random.choice(
        [
            action
            for action, child in node.children.items()
            if ucb_score(config, node, child, min_max_stats) == max_ucb
        ]
    )
    return action, node.children[action]


def ucb_score(
    config: Conf, parent: Node, child: Node, min_max_stats: MinMaxStats
) -> float:
    """."""
    pb_c = (
        math.log((parent.n_k + config.PB_C_BASE + 1) / config.PB_C_BASE)
        + config.PB_C_INIT
    )
    pb_c *= math.sqrt(parent.n_k) / (child.n_k + 1)

    prior_score = pb_c * child.prior
    value_score = min_max_stats.normalize(child.value())
    return prior_score + value_score


def backpropagate(search_path: list[Node], value: float, to_play: bool, discount: float, min_max_stats: MinMaxStats):
    """Backpropagate all the way up the tree to the root."""
    for node in search_path: 
        node.w_k += value if node.to_play == to_play else -value
        node.n_k += 1
        print("")
        print("node v", node.value())
        print("node n_k", node.n_k)
        print("node w_k", node.w_k)
        min_max_stats.update(node.value())

        value = node.reward + discount * value

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
    action_state,
    history: ActionHistory,
    model: M0Net,
    add_exploration_noise=False,
):
    """Runs the Monte-Carlo Tree Search."""
    min_max_stats = MinMaxStats()

    for i in tqdm(range(config.NUM_ROLLOUTS)):
        node, s, history = (
            root_node,
            copy.deepcopy(action_state),
            copy.deepcopy(history),
        )
        search_path = [node]

        # (1) Select
        while not node.is_leaf_node():
            action, node = select_child(config, node, min_max_stats)
            history.add_action(action)
            search_path.append(node)

        parent = search_path[-2]
        # Encode the action for using it as input to the neural net.
        source, t, pro = s.decode_action(history.last_action())
        a = s.encode_single_action(source[0], t[0], pro[0])

        v, r, p, s_next = model.recurrent_inference(parent.hidden_state, a)

        sr = transform_to_scalar(config, r).item()
        sv = transform_to_scalar(config, v).item()

        print("value, reward", sv, sr)
        # 
        # TODO: s.update() in some sort.
        #
        print(node)
        # (2) Expand
        node.expand(s.get_actions(), s.to_play(), sr, p, s_next)

        # if not node.is_leaf_node() or not s.is_terminal():
        #     node = policy(node)

#         # (3) Simulate
#         while not s.is_terminal():
#             s = simulate_policy(s)
#         delta = s.eval()

        # (4) Backpropagate
        backpropagate(search_path, sv, s.to_play(), config.DISCOUNT, min_max_stats)

        # while True:
        #     node.propagate(delta)
        #     if node.is_root_node():
        #         break

        #     node = node.parent

    # return choose_move(root_node)
    return root_node
