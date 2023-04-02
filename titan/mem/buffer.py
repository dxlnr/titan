"""Replay Buffer"""
from titan.config import Conf
from titan.game import Chess


class ReplayBuffer:
    def __init__(self, config: Conf):
        self.config = config
        self.window_size = config.WINDOW_SIZE
        self.batch_size = config.BATCH_SIZE
        self.buffer = []

    def save_game(self, game: Chess):
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self, num_unroll_steps: int, td_steps: int):
        pass
