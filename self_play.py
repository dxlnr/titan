"""Self-Play Procedure"""
from titan.config import Conf
from titan.mcts.chess_state import Chess
from titan.mcts.node import Node
from titan.mem import ReplayBuffer, SharedStorage
from titan.models import M0Net


class ActionHistory:
    """History container used to keep track of the actions executed."""

    def __init__(self):
        self.history = list()
        # self.action_space_size = action_space_size

    def clone(self):
        """."""
        return ActionHistory()

    def add_action(self, action):
        """Appends an action."""
        self.history.append(action)

    def last_action(self):
        """Returns the last action."""
        pass

    def action_space(self) -> list:
        """Returns a list of all actions performed."""
        pass

    def to_play(self):
        pass


def run_selfplay(
    config: Conf, storage: SharedStorage, replay_buffer: ReplayBuffer
):
    model = M0Net(config)

    while True:
        # model = storage.get_latest_model() 
        game = play_game(config, model)
        replay_buffer.save_game(game)

        break


def play_game(config: Conf, model: M0Net):
    """One game is played using the current network parameter and saved to memory.

    Each game is produced by starting at the initial board position, 
    then repeatedly executing a Monte Carlo Tree Search to generate moves 
    until the end of the game is reached.
    """
    game = Chess()
    action_history = ActionHistory()

    while not game.is_terminal() and len(action_history.history) < config.MAX_MOVES:
        # At the root of the search tree we use the representation function to
        # obtain a hidden state given the current observation.
        root = Node()
        current_observation = game.make_image(-1)
        expand_node(root, game.to_play(), game.legal_actions(),
                    model.initial_inference(current_observation))
        add_exploration_noise(config, root)

        # We then run a Monte Carlo Tree Search using only action sequences and the
        # model learned by the network.
        run_mcts(config, root, game.action_history(), model)
        action = select_action(config, len(game.history), root, model)
        game.apply(action)
        game.store_search_statistics(root)
    return game



def main():
    cfg = Conf()
    storage = SharedStorage(cfg)
    buffer = ReplayBuffer()

    run_selfplay(cfg, storage, buffer)


if __name__ == "__main__": main()
