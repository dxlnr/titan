import chess
import random


class State:
    """Defines the state of the specific game. In this case it is chess."""

    def __init__(self, obj):
        # In this case the chess.Board class.
        self.obj = obj

    def get_possible_moves(self) -> list:
        return list(self.obj.legal_moves)

    def is_terminated(self) -> bool:
        return self.obj.is_game_over()

    def update(self, move: str) -> None:
        """Updates the board with a specfic move."""
        self.obj.push(chess.Move.from_uci(str(move)))

    def random_upd(self) -> None:
        """Finds a random move and updates the board."""
        self.update(random.choice(list(self.obj.legal_moves)))

    def eval(self):
        outcome = self.obj.outcome()
        print("")
        print("Game result : ")
        print(outcome)
        print(outcome.result())
        print(outcome.winner)
        delta = 1 if outcome.winner is True else 0
        print("delta : ", delta)
        print("")
        return delta
        

