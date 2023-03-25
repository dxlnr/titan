"""Predict chess moves using MCTS."""
import sys
from pathlib import Path

# Adds root directory to path.
sys.path.append(str(Path(__file__).parent.parent))

import chess

from titan.mcts import mcts
from titan.mcts.chess_state import Chess
from titan.models.muzero import compute_repr


if __name__ == "__main__":
    # Initiate some board state.
    state = Chess(chess.Board())
    s = state.encode_epd() 
    print(s)
    state.encode_board_state()
    
    state.update("e2e4")
    state.update("e7e5")
    state.update("g1f3")
    state.update("b8c6")
    state.update("f1c4")
    state.update("g8f6")
    state.update("f3g5")
    state.update("d7d5")

    print(state.t)
    print(state.state.epd())

    dec1 = state.decode_board_state(timestep=1)
    dec3 = state.decode_board_state(timestep=3)
    dec7 = state.decode_board_state(timestep=7)
    print("")
    print(dec1)
    print("")
    print(dec3)
    print("")
    print(dec7)
    # h = compute_repr(state)

    # print(h)
