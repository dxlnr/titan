"""Predict chess moves using MCTS."""
import sys
from pathlib import Path

# Adds root directory to path.
sys.path.append(str(Path(__file__).parent.parent))

from titan.game.chess_state import Chess
from titan.game.chess_board import Board

if __name__ == "__main__":
    # Initiate some board state.
    state = Chess(Board())
    print(state.state.current_board)
    state.encode_board_state()
    
    print("")
    d1 = state.decode_board_state()
    print(d1)



# import chess

# from titan.mcts import mcts
# from titan.mcts.chess_state import Chess
# # from titan.models.muzero import compute_repr


# if __name__ == "__main__":
#     # Initiate some board state.
#     state = Chess(chess.Board())
#     s = state.encode_epd() 
#     print(s)
#     state.encode_board_state()
    
#     state.update("e2e4")
#     state.update("e7e5")
#     state.update("g1f3")
#     state.update("b8c6")
#     state.update("f1c4")
#     state.update("g8f6")
#     state.update("f3g5")
#     state.update("d7d5")

#     print(state.t)
#     print(state.state.epd())

#     dec1 = state.decode_board_state(timestep=1)
#     dec3 = state.decode_board_state(timestep=3)
#     dec7 = state.decode_board_state(timestep=7)
#     print("")
#     print(dec1)
#     print("")
#     print(dec3)
#     print("")
#     print(dec7)

#     print("")
#     print(state.state.legal_moves)
#     print(list(state.state.legal_moves)[0].from_square)
#     m = list(state.state.legal_moves)[0]
#     print(type(m))
#     print(m)
#     state.encode_board_action(m)
#     # h = compute_repr(state)

#     # print(h)
