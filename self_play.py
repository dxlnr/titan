"""Self-Play Procedure"""
import torch

from titan.config import Conf
# from titan.mcts.chess_state import Chess
from titan.game.chess_state import Chess
from titan.mcts.node import Node
from titan.mcts.action import ActionHistory
from titan.mcts import run_mcts
from titan.mcts.mcts import select_action
from titan.mem import ReplayBuffer, SharedStorage
from titan.models import M0Net


def run_selfplay(config: Conf, storage: SharedStorage, replay_buffer: ReplayBuffer):
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
    observation = game.get_observation()
    # initialize the Action state history.
    action_history.action_history.append(0)
    action_history.observation_history.append(observation)
    action_history.reward_history.append(0)
    action_history.to_play_history.append(game.to_play())

    with torch.no_grad():
        while (
            not game.is_terminal()
            and len(action_history.action_history) < config.MAX_MOVES
        ):
            # At the root of the search tree we use the representation function to
            # obtain a hidden state given the current observation.
            root = Node(0)
            # 
            if len(observation.shape) == 3:
                observation = observation.float().unsqueeze(0).to(next(model.parameters()).device)
            else:
                observation = observation.float().to(next(model.parameters()).device)
 
            print(observation.shape)
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
            run_mcts(config, root, game, action_history, model)
            # root_node = run_mcts(config, root, action_state, model)

            # action = select_action(root)
            # game.update(str(action))
            # print(str(action))
            # game.store_search_statistics(root)
    return game


def main():
    cfg = Conf()
    storage = SharedStorage(cfg)
    buffer = ReplayBuffer()

    run_selfplay(cfg, storage, buffer)


if __name__ == "__main__":
    main()
