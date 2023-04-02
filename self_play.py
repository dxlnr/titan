"""Self-Play Procedure"""
import torch

from titan.config import Conf
from titan.game import Chess
from titan.mcts import run_mcts, select_action
from titan.mcts.action import ActionHistory
from titan.mcts.node import Node
from titan.mem import ReplayBuffer, SharedStorage
from titan.models import M0Net
from titan.models.muzero import transform_to_scalar


def run_selfplay(
    config: Conf, storage: SharedStorage, replay_buffer: ReplayBuffer
) -> None:
    model = M0Net(config)

    while True:
        # model = storage.get_latest_model()
        game = play_game(config, model)
        replay_buffer.save_game(game)
        break


def play_game(config: Conf, model: M0Net) -> Chess:
    """One game is played using the current network parameter and saved to memory.

    Each game is produced by starting at the initial board position,
    then repeatedly executing a Monte Carlo Tree Search to generate moves
    until the end of the game is reached.

    :param config: Custom Configuration.
    :param model: Muzero Neural Network guiding the search.
    """
    game = Chess(config)
    # initialize the Action state history.
    game.history.append(0)
    game.reward.append(0)
    
    with torch.no_grad():
        while not game.is_terminal() and len(game.history) < config.MAX_MOVES:
            # At the root of the search tree we use the representation function to
            # obtain a hidden state given the current observation.
            root = Node(0)

            observation = game.get_observation()
            #
            if len(observation.shape) == 3:
                observation = (
                    observation.float().unsqueeze(0).to(next(model.parameters()).device)
                )
            else:
                observation = observation.float().to(next(model.parameters()).device)

            (
                root_predicted_value,
                reward,
                policy,
                hidden_state,
            ) = model.initial_inference(observation)

            sr = transform_to_scalar(config, reward).item()
            root.expand(game.get_actions(), game.to_play(), sr, policy, hidden_state)
            root.add_exploration_noise(config)

            # We then run a Monte Carlo Tree Search using only action sequences and the
            # model learned by the network.
            root = run_mcts(config, root, game, model)

            action = select_action(root)
            # Update the game.
            game.apply(action)
            game.store_search_stats(config, root)

            print(game.state.current_board)

    return game


def main():
    cfg = Conf()
    print(cfg)
    storage = SharedStorage(cfg)
    buffer = ReplayBuffer(cfg)

    run_selfplay(cfg, storage, buffer)


if __name__ == "__main__":
    main()
