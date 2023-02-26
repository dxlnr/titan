"""Application Endpoint"""
import os

from flask import Flask
from flask import render_template

from titan.utils import get_project_root

# Chess Engine Server
app = Flask(__name__)

@app.route("/")
def chess_server():
    b = open(os.path.join(get_project_root(), "index.html")).read()
    print(type(b))
    # return b.replace('start', s.board.fen())
    # return render_template("index.html")
    return b


if __name__ == "__main__":
    app.run(host="localhost", port=5000, debug=True)
