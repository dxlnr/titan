class State:
    """Defines the state of the specific game. In this case it is chess."""

    def __init__(self, obj):
        # In this case the chess.Board class.
        self.obj = obj

    def get_possible_moves(self):
        return list(self.obj.legal_moves)

    def is_terminated(self) -> bool:
        return self.obj.is_game_over()
