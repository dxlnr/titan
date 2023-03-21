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


@app.route("/move", methods=["GET", "POST"])
def move():
    fen = request.form["new_pos"]
    s.state.set_fen(fen)
    s.state.turn = not s.state.turn
    
    print(s.state.turn)
    print(len(list(s.state.legal_moves)))
    print(list(s.state.legal_moves))
    return jsonify(success=True)


@app.route("/engine_move", methods=["GET"])
def engine_move():
    # print(list(s.state.legal_moves))
    move = mcts(s)
    # test_move = "e7e5"
    s.state.push(chess.Move.from_uci(str(move)))

    return jmove(move)


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
