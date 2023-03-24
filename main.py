"""Playing Chess against an Engine."""
import os
import chess
import copy

from flask import Flask, request, jsonify
from flask import render_template

from titan.utils import get_project_root
from titan.mcts import mcts
from titan.mcts.chess_state import ChessState

# Chess Engine Server
app = Flask(__name__)


def jmove(move: chess.Move) -> str:
    """Prepares Move object from python-chess for the ui."""
    # Gets a UCI string for the move.
    m = move.uci()
    # Chessboardjs demands a dash.
    return m[:2] + "-" + m[2:]


@app.route("/get_move", methods=["GET", "POST"])
def get_move():
    """Reads a move from input."""
    fen = request.form["new_pos"]
    s.state.set_fen(fen)
    print(list(s.state.legal_moves))

    return jsonify(success=True)


@app.route("/engine_move", methods=["GET"])
def engine_move():
    """Runs the model and returns a legal move."""
    move = mcts(s, 500)
    s.state.push(chess.Move.from_uci(str(move)))

    return {
        "source": chess.square_name(move.from_square),
        "target": chess.square_name(move.to_square),
    }


@app.route("/")
def chess_server():
    # Access root index html.
    b = open(os.path.join(get_project_root(), "index.html")).read()
    # Return board with a specific board state.
    return b.replace("position: 'start'", f"position: '{s.state.fen()}'")


if __name__ == "__main__":
    global s
    s = ChessState(chess.Board())

    # Run the server.
    app.run(host="localhost", port=5000, debug=True)
