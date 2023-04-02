"""Monte-Carlo Tree Search"""
import copy
import math

import numpy as np
from tqdm import tqdm

from titan.config import Conf
from titan.mcts.node import Node
from titan.models import M0Net
from titan.models.muzero import transform_to_scalar


class MinMaxStats:
    """A class that holds the min-max values of the tree."""

    MAXIMUM_FLOAT_VALUE: float = float("inf")

    def __init__(self):
        self.maximum = -self.MAXIMUM_FLOAT_VALUE
        self.minimum = self.MAXIMUM_FLOAT_VALUE

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


def select_action(node: Node, temperature: float = 0) -> int:
    """Select an action based on temperature value. Returns an index value."""
    vc = np.array([child.n_k for child in node.children.values()])
    actions = [action for action in node.children.keys()]

    if temperature == 0:
        action = actions[np.argmax(vc)]
    elif temperature == float("inf"):
        action = np.random.choice(actions)
    else:
        v_dist = vc ** (1 / temperature)
        v_dist = v_dist / sum(v_dist)
        action = np.random.choice(actions, p=v_dist)

    return action


def select_child(config: Conf, node: Node, min_max_stats: MinMaxStats):
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


def backpropagate(
    search_path: list[Node],
    value: float,
    to_play: bool,
    discount: float,
    min_max_stats: MinMaxStats,
):
    """Backpropagate all the way up the tree to the root."""
    for node in search_path:
        node.w_k += value if node.to_play == to_play else -value
        node.n_k += 1
        min_max_stats.update(node.value())

        value = node.reward + discount * value


def run_mcts(
    config: Conf,
    root_node: Node,
    action_state,
    model: M0Net,
    add_exploration_noise=False,
):
    """Runs the Monte-Carlo Tree Search."""
    min_max_stats = MinMaxStats()

    for i in tqdm(range(config.NUM_ROLLOUTS)):
        node, s = (
            root_node,
            copy.deepcopy(action_state),
        )
        history = s.action_history()
        search_path = [node]
        # Select
        while not node.is_leaf_node():
            action, node = select_child(config, node, min_max_stats)
            history.add_action(action)
            search_path.append(node)

        parent = search_path[-2]
        # Encode the action for using it as input to the neural net.
        source, t, pro = s.decode_action(history.last_action())
        a = s.encode_single_action(source[0], t[0], pro[0])
        #
        v, r, p, s_next = model.recurrent_inference(parent.hidden_state, a)
        # Transform the distributions to scalars.
        sr = transform_to_scalar(config, r).item()
        sv = transform_to_scalar(config, v).item()
        # Expand
        node.expand(config.ACTION_SPACE, s.to_play(), sr, p, s_next)
        # Backpropagate
        backpropagate(search_path, sv, s.to_play(), config.DISCOUNT, min_max_stats)

    return search_path[0]
