import chess
import random


class State:
    """Defines the state of the specific game. In this case it is chess."""

    def __init__(self, obj):
        # In this case the chess.Board class.
        self.obj = obj

    def get_possible_moves(self):
        return list(self.obj.legal_moves)

    def is_terminated(self) -> bool:
        return self.obj.is_game_over()

    def update(self, move: str):
        self.obj.push(chess.Move.from_uci(str(move)))

    def random_upd(self):
        # lm = list(self.obj.legal_moves)
        self.update(random.choice(list(self.obj.legal_moves)))

    def eval(self):
        outcome = self.obj.outcome
        print("")
        print("WINNNNNNNER?")
        print(outcome)
        print("")
        # delta = 1 if outcome.winner 

