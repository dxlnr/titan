class State:
    """Defines the state of the specific game. In this case it is chess."""

    def __init__(self, obj):
        # In this case the chess.Board class.
        self.obj = obj

    def get_possible_moves(self):
        return list(self.obj.legal_moves)

    def __getitem__(self, idx):
        pass
    
    @classmethod
    def get_updated(cls, move: str):
        new_state = cls.copy()
        # move = chess.Move.from_uci(move)
        # print(move)
        new_state.state.push(chess.Move.from_uci(move))
        # return board.push(chess.Move.from_uci(move))
        return new_state
