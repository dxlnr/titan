"""State Representation of Chess."""
import random

import chess
import torch

from titan.mcts.state import State
from titan.utils.helper import chunks


class Chess(State):
    """Defines the state of a chess game."""

    # Board Dimension
    N = 8
    # M feature indicating the presence of the player’s pieces,
    M = 12 + 2
    # Additional L constant-valued input planes denoting the player’s colour,
    # the total move count, and the state of special rules.
    L = 7
    # Chess Pieces Mapping
    MAP = {
        "R": 0,  # White Rook
        "N": 1,  # White Knight
        "B": 2,  # White Bishop
        "Q": 3,  # White Queen
        "K": 4,  # White King
        "P": 5,  # White Pawn
        "r": 6,  # Black Rook
        "n": 7,  # Black Knight
        "b": 8,  # Black Bishop
        "q": 9,  # Black Queen
        "k": 10,  # Black King
        "p": 11,  # Black Pawn
    }

    def __init__(self, state=None):
        # In this case the chess.Board class.
        if state is None:
            self.state = chess.Board()
        else:
            self.state = state

        # Timesteps, history and current
        self.T, self.t = 8, 0
        # Representation of the board inputs which gets feeded to h (representation).
        self.enc = torch.zeros([self.N, self.N, (self.M * self.T + self.L + 1)])

        self.c_w_epd = None
        self.c_b_epd = None
        self.w_repetitions = 0
        self.b_repetitions = 0
        self.no_progress_count = 0

    def encode_epd(self) -> list:
        """Convert an edp notation to a list representing a board state."""
        res = list()
        for c in self.state.epd():
            if c == " ":
                return res
            elif c != "/":
                if c in self.MAP:
                    res.append(self.MAP[c])
                else:
                    [res.append(None) for counter in range(int(c))]
        return res

    def decode_to_epd(self) -> str:
        """Decodes the current board state back to an edp notation."""
        pass

    def encode_board_state(self) -> torch.Tensor:
        """Encodes the board state according to MuZero paper."""
        board = self.encode_epd()

        # Roll over the tensor to inject the latest position from time step t.
        self.enc = torch.roll(self.enc, -self.M, 2)
        self.enc[:, :, ((self.T - 1) * self.M) :] = 0

        for i, r in enumerate(chunks(board, 8)):
            for j, c in enumerate(r):
                if c is not None:
                    self.enc[i, j, c + ((self.T - 1) * self.M)] = 1

        if self.state.turn == True:
            self.enc[:, :, 112] = 1
        # Castling
        if self.state.has_kingside_castling_rights(chess.WHITE) == True:
            self.enc[:, :, 113] = 1  # can castle kingside for white
        if self.state.has_queenside_castling_rights(chess.WHITE) == True:
            self.enc[:, :, 114] = 1  # can castle queenside for white
        if self.state.has_kingside_castling_rights(chess.BLACK) == True:
            self.enc[:, :, 115] = 1  # can castle kingside for black
        if self.state.has_queenside_castling_rights(chess.BLACK) == True:
            self.enc[:, :, 116] = 1  # can castle queenside for black

        self.enc[:, :, 117] = self.state.ply()
        # This denotes the progress count. Will be for now set to zero as
        # the history parameter T is only 8 so it is not possible to keep state
        # of the 50 move rule.
        self.enc[:, :, 118] = self.no_progress_count
        self.enc[:, :, 119] = 1 if self.state.has_legal_en_passant() else 0

    def decode_board_state(self, timestep: int = 0):
        """."""
        assert timestep <= self.t, f"Input timestep {timestep} is in the future."
        # Define the offset.
        offset = self.t - timestep
        assert offset <= 7, f"Timestep lies too far in the past."

        dec = [None] * self.N**2
        inv_map = {v: k for k, v in self.MAP.items()}

        counter = 0
        for i in range(self.N):
            for j in range(self.N):
                for k in range(self.M - 2):
                    if self.enc[i, j, k + ((self.T - 1 - offset) * self.M)] == 1:
                        dec[counter] = k
                counter += 1
        return dec

    def reset(self):
        """Resets game state and all its internal attributes."""
        pass

    def get_possible_moves(self) -> list:
        return list(self.state.legal_moves)

    def is_terminal(self) -> bool:
        return self.state.is_game_over()

    def update(self, move: str) -> None:
        """Updates the board with a specfic move."""
        # Sets the current game state before updating it.
        if self.state.turn:
            self.c_w_epd = self.state.epd()
        else:
            self.c_b_epd = self.state.epd()

        # Update the board state with new move.
        self.state.push(chess.Move.from_uci(str(move)))

        # Count up if repetition happend.
        if self.state.turn:
            self.w_repetitions = (
                self.w_repetitions + 1 if self.c_w_epd == self.state.epd() else 0
            )
        else:
            self.b_repetitions = (
                self.b_repetitions + 1 if self.c_b_epd == self.state.epd() else 0
            )

        # Update internal timestep
        self.t = self.t + 1
        # Encode with latest board state.
        self.encode_board_state()

    def random_upd(self) -> None:
        """Finds a random move and updates the board."""
        self.update(random.choice(list(self.state.legal_moves)))

    def eval(self):
        """."""
        outcome = self.state.outcome()
        delta = 1 if outcome.winner is True else 0
        return delta
