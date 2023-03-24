"""Self-Play Procedure"""


def run_selfplay(
    config: MuZeroConfig, storage: SharedStorage, replay_buffer: ReplayBuffer
):
    while True:
        network = storage.latest_network()
        game = play_game(config, network)
        replay_buffer.save_game(game)
