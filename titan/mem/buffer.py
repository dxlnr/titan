"""Replay Buffer"""
from titan.config import Conf
from titan.game import Chess


class ReplayBuffer:

    def __init__(self, config: Conf):
        self.config = config
        self.window_size = self.config.WINDOW_SIZE
        self.batch_size = self.config.BATCH_SIZE
        self.buffer = []

    def save_game(self, game: Chess):
        """Save game."""
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self, num_unroll_steps: int, td_steps: int):
        """Sample a batch for training the muzero model."""
        games = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.sample_position(g)) for g in games]
        return [
            (
                g.get_observation(i),
                g.history[i : i + num_unroll_steps],
                g.make_target(i, num_unroll_steps, td_steps, g.to_play()),
            )
            for (g, i) in game_pos
        ]

    def sample_game(self) -> Chess:
        """Sample game from buffer either uniformly or according to some priority."""
        return self.buffer[0]

    def sample_position(self, game) -> int:
        """Sample position from game either uniformly or according to some priority."""
        return -1
