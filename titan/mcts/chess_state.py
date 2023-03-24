"""State Representation of Chess."""
import random

import chess
import torch

from titan.mcts.state import State
from titan.utils.helper import chunks


class Chess(State):
    """Defines the state of a chess game."""

    mapped = {
        "R": 0,  # White Rook
        "N": 1,  # White Knight
        "B": 2,  # White Bishop
        "Q": 3,  # White Queen
        "K": 4,  # White King
        "P": 5,  # White Pawn
        "r": 6,
        "n": 7,
        "b": 8,
        "q": 9,
        "k": 10,
        "p": 11,
    }

    def __init__(self, state=None):
        # In this case the chess.Board class.
        if state is None:
            self.state = chess.Board()
        else:
            self.state = state

        # self.enc = self.encode_state()

    def convert_epd(self) -> list:
        """Convert an edp notation to a list representing a board state."""
        res = list()
        for c in self.state.epd():
            if c == " ":
                return res
            elif c != "/":
                if c in self.mapped:
                    res.append(self.mapped[c])
                else:
                    [res.append(None) for counter in range(int(c))]
        return res

    def encode_board(self) -> torch.Tensor:
        """."""
        enc = torch.zeros([8, 8, 22])
        board = self.convert_epd()

        for i, r in enumerate(chunks(board, 8)):
            for j, c in enumerate(r):
                if c is not None:
                    enc[i, j, c] = 1

        if self.state.turn == True:
            enc[:, :, 12] = 1
        # Castling
        if self.state.has_kingside_castling_rights(chess.WHITE) == True:
            enc[:, :, 13] = 1  # can castle kingside for white
        if self.state.has_queenside_castling_rights(chess.WHITE) == True:
            enc[:, :, 14] = 1  # can castle queenside for white
        if self.state.has_kingside_castling_rights(chess.BLACK) == True:
            enc[:, :, 15] = 1  # can castle kingside for black
        if self.state.has_queenside_castling_rights(chess.BLACK) == True:
            enc[:, :, 16] = 1  # can castle queenside for black

        enc[:, :, 17] = self.state.fullmove_number
        # enc[:, :, 18] = self.state.repetitions_w
        # enc[:, :, 19] = board.repetitions_b
        # enc[:, :, 20] = board.no_progress_count
        enc[:, :, 21] = 1 if self.state.has_legal_en_passant() else 0
        return enc

    def get_possible_moves(self) -> list:
        return list(self.state.legal_moves)

    def is_terminal(self) -> bool:
        return self.state.is_game_over()

    def update(self, move: str) -> None:
        """Updates the board with a specfic move."""
        self.state.push(chess.Move.from_uci(str(move)))

    def random_upd(self) -> None:
        """Finds a random move and updates the board."""
        self.update(random.choice(list(self.state.legal_moves)))

    def eval(self):
        """."""
        outcome = self.state.outcome()
        delta = 1 if outcome.winner is True else 0
        return delta
