"""Application Endpoint"""
import os
import chess
import copy

from flask import Flask, request, jsonify
from flask import render_template

from titan.utils import get_project_root
from titan.mcts.state import State

# Chess Engine Server
app = Flask(__name__)


@app.route("/move", methods=["GET", "POST"])
def move():
    fen = request.form["new_pos"]
    s.obj.set_fen(fen)

    return jsonify(success=True)


@app.route("/engine_move", methods=["GET"])
def engine_move():
    test_move = "e7e5"
    s.obj.push(chess.Move.from_uci(str(test_move)))

    return test_move[:2] + "-" + test_move[2:]


@app.route("/")
def chess_server():
    # Access root index html.
    b = open(os.path.join(get_project_root(), "index.html")).read()
    # Return board with a specific board state.
    return b.replace("position: 'start'", f"position: '{s.obj.fen()}'")


if __name__ == "__main__":
    global s
    s = State(chess.Board())

    # Run the server.
    app.run(host="localhost", port=5000, debug=True)
