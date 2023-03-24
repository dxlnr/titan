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
    s = state.convert_epd() 
    print(s)
    state.encode_board()
 
    # h = compute_repr(state)

    # print(h)
