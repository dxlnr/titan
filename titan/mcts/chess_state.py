"""State Representation of Chess."""
import random
import chess

from titan.mcts.state import State


class ChessState(State):
    """Defines the state of a chess game."""

    def __init__(self, state=None):
        # In this case the chess.Board class.
        if state is None:
            self.state = chess.Board()
        else:
            self.state = state

    def get_possible_moves(self) -> list:
        return list(self.state.legal_moves)

    def is_terminal(self) -> bool:
        return self.state.is_game_over()

    def update(self, move: str) -> None:
        """Updates the board with a specfic move."""
        self.state.push(chess.Move.from_uci(str(move)))

    def random_upd(self) -> None:
        """Finds a random move and updates the board."""
        self.update(random.choice(list(self.state.legal_moves)))

    def eval(self):
        """."""
        outcome = self.state.outcome()
        delta = 1 if outcome.winner is True else 0
        return delta
