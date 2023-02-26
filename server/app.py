"""Chess Engine Server"""
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from flask import Flask
from titan.utils import get_project_root


app = Flask(__name__)

@app.route("/")
def chess_server():
    b = open(os.path.join(get_project_root(), "index.html")).read()
    # return b.replace('start', s.board.fen())
    return b


app.run(host="localhost", port=5000, debug=True)
