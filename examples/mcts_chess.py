"""Predict chess moves using MCTS."""
import sys
from pathlib import Path

# Adds root directory to path.
sys.path.append(str(Path(__file__).parent.parent))

import chess

from titan.mcts import mcts
from titan.mcts.chess_state import ChessState


if __name__ == "__main__":
    # Initiate some board state.
    state = ChessState(chess.Board())
    
    # Runs the Monte-Carlo Tree Search to predict the next move.
    move = mcts(state)
    
    # Print results.
    print("")
    print("Predicted move : ", move)
    print("")
