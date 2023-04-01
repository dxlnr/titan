"""Chess Game State"""
import numpy as np
import torch

from titan.mcts.state import State
from titan.game.chess_board import Board


class Chess():
    """Defines the state of a chess game."""

    # Board Dimension
    N = 8
    # M feature indicating the presence of the player’s pieces,
    M = 12 + 2
    # Additional L constant-valued input planes denoting the player’s colour,
    # the total move count, and the state of special rules.
    L = 7
    # 73 target square possibilities 
    # (NRayDirs x MaxRayLength + NKnightDirs + NPawnDirs * NMinorPromotions), 
    # encoding a probability distribution over 64x73 = 4,672 possible moves
    TS = 73
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
            self.state = Board()
        else:
            self.state = state
        
        # Timesteps, history and current
        self.T, self.t = 8, 0
        # Representation of the board inputs which gets feeded to h (representation).
        # C, H, W : 119 x 8 x 8
        self.enc_state = torch.zeros([(self.M * self.T + self.L), self.N, self.N])

        # Add the initial position as encoded board state.
        self.encode_board_state()
        
    def encode_board_state(self) -> None:
        """."""
        board_state = self.state.current_board; 
        
        # Roll over the tensor to inject the latest position from time step t.
        self.enc_state = torch.roll(self.enc_state, -self.M, 0)
        self.enc_state[((self.T - 1) * self.M) :, :, :] = 0

        for i in range(self.N):
            for j in range(self.N):
                if board_state[i,j] != " ":
                    self.enc_state[self.MAP[board_state[i, j]] + ((self.T - 1) * self.M), i, j] = 1
        
        # Repetitions for each side.
        self.enc_state[110, :, :] = self.state.repetitions_w
        self.enc_state[111, :, :] = self.state.repetitions_b
        # Color
        if self.state.player == 1:
            self.enc_state[112, :, :] = 1
        # Castling
        if self.state.K_move_count == 0 and self.state.R1_move_count == 0:
            self.enc_state[113, : , :] = 1 # can castle kingside for white
        if self.state.K_move_count == 0 and self.state.R2_move_count == 0:
            self.enc_state[114, :, :] = 1  # can castle queenside for white
        if self.state.k_move_count == 0 and self.state.r1_move_count == 0:
            self.enc_state[115, :, :] = 1 # can castle kingside for black
        if self.state.k_move_count == 0 and self.state.r2_move_count == 0:
            self.enc_state[116, :, :] = 1 # can castle queenside for black
        # Total Move Count
        self.enc_state[117, :, :] = self.state.move_count
        # This denotes the progress count.
        self.enc_state[118, :, :] = self.state.no_progress_count

    def decode_board_state(self, timestep: int = 0):
        """."""
        assert timestep <= self.t, f"Input timestep {timestep} greater than {self.t}."
        # Define the offset.
        offset = self.t - timestep
        assert offset <= 7, f"Timestep lies too far in the past."

        dec = np.zeros([8,8]).astype(str)
        dec[dec == "0.0"] = " "
        inv_map = {v: k for k, v in self.MAP.items()}

        for i in range(self.N):
            for j in range(self.N):
                for k in range(self.M - 2):
                    if self.enc_state[k + ((self.T - 1 - offset) * self.M), i, j] == 1:
                        dec[i,j] = inv_map[k]
        return dec

#     def encode_action(board,initial_pos,final_pos,underpromote=None):
#         encoded = np.zeros([8,8,73]).astype(int)
#         i, j = initial_pos; x, y = final_pos; dx, dy = x-i, y-j
#         piece = board.current_board[i,j]
#         if piece in ["R","B","Q","K","P","r","b","q","k","p"] and underpromote in [None,"queen"]: # queen-like moves
#             if dx != 0 and dy == 0: # north-south idx 0-13
#                 if dx < 0:
#                     idx = 7 + dx
#                 elif dx > 0:
#                     idx = 6 + dx
#             if dx == 0 and dy != 0: # east-west idx 14-27
#                 if dy < 0:
#                     idx = 21 + dy
#                 elif dy > 0:
#                     idx = 20 + dy
#             if dx == dy: # NW-SE idx 28-41
#                 if dx < 0:
#                     idx = 35 + dx
#                 if dx > 0:
#                     idx = 34 + dx
#             if dx == -dy: # NE-SW idx 42-55
#                 if dx < 0:
#                     idx = 49 + dx
#                 if dx > 0:
#                     idx = 48 + dx
#         if piece in ["n","N"]: # Knight moves 56-63
#             if (x,y) == (i+2,j-1):
#                 idx = 56
#             elif (x,y) == (i+2,j+1):
#                 idx = 57
#             elif (x,y) == (i+1,j-2):
#                 idx = 58
#             elif (x,y) == (i-1,j-2):
#                 idx = 59
#             elif (x,y) == (i-2,j+1):
#                 idx = 60
#             elif (x,y) == (i-2,j-1):
#                 idx = 61
#             elif (x,y) == (i-1,j+2):
#                 idx = 62
#             elif (x,y) == (i+1,j+2):
#                 idx = 63
#         if piece in ["p", "P"] and (x == 0 or x == 7) and underpromote != None: # underpromotions
#             if abs(dx) == 1 and dy == 0:
#                 if underpromote == "rook":
#                     idx = 64
#                 if underpromote == "knight":
#                     idx = 65
#                 if underpromote == "bishop":
#                     idx = 66
#             if abs(dx) == 1 and dy == -1:
#                 if underpromote == "rook":
#                     idx = 67
#                 if underpromote == "knight":
#                     idx = 68
#                 if underpromote == "bishop":
#                     idx = 69
#             if abs(dx) == 1 and dy == 1:
#                 if underpromote == "rook":
#                     idx = 70
#                 if underpromote == "knight":
#                     idx = 71
#                 if underpromote == "bishop":
#                     idx = 72
#         encoded[i,j,idx] = 1
#         encoded = encoded.reshape(-1); encoded = np.where(encoded==1)[0][0] #index of action
#         return encoded

#     def decode_action(board,encoded):
#         encoded_a = np.zeros([4672]); encoded_a[encoded] = 1; encoded_a = encoded_a.reshape(8,8,73)
#         a,b,c = np.where(encoded_a == 1); # i,j,k = i[0],j[0],k[0]
#         i_pos, f_pos, prom = [], [], []
#         for pos in zip(a,b,c):
#             i,j,k = pos
#             initial_pos = (i,j)
#             promoted = None
#             if 0 <= k <= 13:
#                 dy = 0
#                 if k < 7:
#                     dx = k - 7
#                 else:
#                     dx = k - 6
#                 final_pos = (i + dx, j + dy)
#             elif 14 <= k <= 27:
#                 dx = 0
#                 if k < 21:
#                     dy = k - 21
#                 else:
#                     dy = k - 20
#                 final_pos = (i + dx, j + dy)
#             elif 28 <= k <= 41:
#                 if k < 35:
#                     dy = k - 35
#                 else:
#                     dy = k - 34
#                 dx = dy
#                 final_pos = (i + dx, j + dy)
#             elif 42 <= k <= 55:
#                 if k < 49:
#                     dx = k - 49
#                 else:
#                     dx = k - 48
#                 dy = -dx
#                 final_pos = (i + dx, j + dy)
#             elif 56 <= k <= 63:
#                 if k == 56:
#                     final_pos = (i+2,j-1)
#                 elif k == 57:
#                     final_pos = (i+2,j+1)
#                 elif k == 58:
#                     final_pos = (i+1,j-2)
#                 elif k == 59:
#                     final_pos = (i-1,j-2)
#                 elif k == 60:
#                     final_pos = (i-2,j+1)
#                 elif k == 61:
#                     final_pos = (i-2,j-1)
#                 elif k == 62:
#                     final_pos = (i-1,j+2)
#                 elif k == 63:
#                     final_pos = (i+1,j+2)
#             else:
#                 if k == 64:
#                     if board.player == 0:
#                         final_pos = (i-1,j)
#                         promoted = "R"
#                     if board.player == 1:
#                         final_pos = (i+1,j)
#                         promoted = "r"
#                 if k == 65:
#                     if board.player == 0:
#                         final_pos = (i-1,j)
#                         promoted = "N"
#                     if board.player == 1:
#                         final_pos = (i+1,j)
#                         promoted = "n"
#                 if k == 66:
#                     if board.player == 0:
#                         final_pos = (i-1,j)
#                         promoted = "B"
#                     if board.player == 1:
#                         final_pos = (i+1,j)
#                         promoted = "b"
#                 if k == 67:
#                     if board.player == 0:
#                         final_pos = (i-1,j-1)
#                         promoted = "R"
#                     if board.player == 1:
#                         final_pos = (i+1,j-1)
#                         promoted = "r"
#                 if k == 68:
#                     if board.player == 0:
#                         final_pos = (i-1,j-1)
#                         promoted = "N"
#                     if board.player == 1:
#                         final_pos = (i+1,j-1)
#                         promoted = "n"
#                 if k == 69:
#                     if board.player == 0:
#                         final_pos = (i-1,j-1)
#                         promoted = "B"
#                     if board.player == 1:
#                         final_pos = (i+1,j-1)
#                         promoted = "b"
#                 if k == 70:
#                     if board.player == 0:
#                         final_pos = (i-1,j+1)
#                         promoted = "R"
#                     if board.player == 1:
#                         final_pos = (i+1,j+1)
#                         promoted = "r"
#                 if k == 71:
#                     if board.player == 0:
#                         final_pos = (i-1,j+1)
#                         promoted = "N"
#                     if board.player == 1:
#                         final_pos = (i+1,j+1)
#                         promoted = "n"
#                 if k == 72:
#                     if board.player == 0:
#                         final_pos = (i-1,j+1)
#                         promoted = "B"
#                     if board.player == 1:
#                         final_pos = (i+1,j+1)
#                         promoted = "b"
#             if board.current_board[i,j] in ["P","p"] and final_pos[0] in [0,7] and promoted == None: # auto-queen promotion for pawn
#                 if board.player == 0:
#                     promoted = "Q"
#                 else:
#                     promoted = "q"
#             i_pos.append(initial_pos); f_pos.append(final_pos), prom.append(promoted)
#         return i_pos, f_pos, prom 

#     def encode_epd(self) -> list:
#         """Convert an edp notation to a list representing a board state."""
#         res = list()
#         for c in self.state.epd():
#             if c == " ":
#                 return res
#             elif c != "/":
#                 if c in self.MAP:
#                     res.append(self.MAP[c])
#                 else:
#                     [res.append(None) for counter in range(int(c))]
#         return res

#     def decode_to_epd(self) -> str:
#         """Decodes the current board state back to an edp notation."""
#         pass

#     def encode_board_state(self) -> torch.Tensor:
#         """Encodes the board state according to MuZero paper."""
#         board = self.encode_epd()

#         # Roll over the tensor to inject the latest position from time step t.
#         self.enc_state = torch.roll(self.enc_state, -self.M, 0)
#         self.enc_state[((self.T - 1) * self.M) :, :, :] = 0

#         for i, r in enumerate(chunks(board, 8)):
#             for j, c in enumerate(r):
#                 if c is not None:
#                     self.enc_state[c + ((self.T - 1) * self.M), i, j] = 1

#         # Repetitions for each side.
#         if self.state.turn:
#             self.enc_state[110, :, :] = self.w_repetitions
#         else:
#             self.enc_state[111, :, :] = self.b_repetitions
#         # Color
#         if self.state.turn == True:
#             self.enc_state[112, :, :] = 1
#         # Castling
#         if self.state.has_kingside_castling_rights(chess.WHITE) == True:
#             self.enc_state[113, :, :] = 1  # can castle kingside for white
#         if self.state.has_queenside_castling_rights(chess.WHITE) == True:
#             self.enc_state[114, :, :] = 1  # can castle queenside for white
#         if self.state.has_kingside_castling_rights(chess.BLACK) == True:
#             self.enc_state[115, :, :] = 1  # can castle kingside for black
#         if self.state.has_queenside_castling_rights(chess.BLACK) == True:
#             self.enc_state[116, :, :] = 1  # can castle queenside for black
#         # Total Move Count
#         self.enc_state[117, :, :] = self.state.ply()
#         # This denotes the progress count. Will be for now set to zero as
#         # the history parameter T is only 8 so it is not possible to keep state
#         # of the 50 move rule.
#         self.enc_state[118, :, :] = self.no_progress_count

#     def decode_board_state(self, timestep: int = 0):
#         """."""
#         assert timestep <= self.t, f"Input timestep {timestep} is in the future."
#         # Define the offset.
#         offset = self.t - timestep
#         assert offset <= 7, f"Timestep lies too far in the past."

#         dec = [None] * self.N**2
#         inv_map = {v: k for k, v in self.MAP.items()}

#         counter = 0
#         for i in range(self.N):
#             for j in range(self.N):
#                 for k in range(self.M - 2):
#                     if self.enc_state[k + ((self.T - 1 - offset) * self.M), i, j] == 1:
#                         dec[counter] = k
#                 counter += 1
#         return dec

#     def encode_board_action(self, move: chess.Move):
#         """."""
#         # Action representation
#         enc_action = torch.zeros([8, self.N, self.N])
#         # Get index from move square.
#         s_i, s_j = divmod(move.from_square, self.N)
#         t_i, t_j = divmod(move.to_square, self.N)

#         # Position the piece was moved from.
#         enc_action[0, s_i, s_j] = 1
#         # Position the piece was moved to.
#         enc_action[1, t_i, t_j] = 1
#         # If the move was legal
#         enc_action[2, :, :] = 1
#         # Promotion encodings.
#         if move.promotion:
#             enc_action[(move.promotion.piece_type + 1), :, :] = 1
#         else:
#             enc_action[7, :, :] = 1

#         return enc_action
    
#     # def encode_actions(self):
#     #     """."""
#     #     board = self.encode_epd()
#     #     # Action representation: 73 x 8 x 8
#     #     enc_action = torch.zeros([self.TS, self.N. self.N])
#     #     # # Get index from move square.
#     #     # s_i, s_j = divmod(move.from_square, self.N)
#     #     # t_i, t_j = divmod(move.to_square, self.N)
        
#     #     for i in range(self.N):
#     #         for j in range(self.N):

#     #             enc_action[


#     def reset(self):
#         """Resets game state and all its internal attributes."""
#         pass

#     def get_observation(self) -> torch.Tensor:
#         """Returns the observation tensor."""
#         return self.enc_state

#     def get_legal_actions(self) -> list:
#         return [str(m) for m in list(self.state.legal_moves)]

#     def is_terminal(self) -> bool:
#         return self.state.is_game_over()

#     def to_play(self) -> bool:
#         """Returns the color that is next up to play."""
#         return self.state.turn

#     def update(self, move: str) -> None:
#         """Updates the board with a specfic move."""
#         # Sets the current game state before updating it.
#         if self.state.turn:
#             self.c_w_epd = self.state.epd()
#         else:
#             self.c_b_epd = self.state.epd()

#         # Update the board state with new move.
#         self.state.push(chess.Move.from_uci(str(move)))

#         # Count up if repetition happend.
#         if self.state.turn:
#             self.w_repetitions = (
#                 self.w_repetitions + 1 if self.c_w_epd == self.state.epd() else 0
#             )
#         else:
#             self.b_repetitions = (
#                 self.b_repetitions + 1 if self.c_b_epd == self.state.epd() else 0
#             )

#         # Update internal timestep
#         self.t = self.t + 1
#         # Encode with latest board state.
#         self.encode_board_state()

#     def random_upd(self) -> None:
#         """Finds a random move and updates the board."""
#         self.update(random.choice(list(self.state.legal_moves)))

#     def eval(self):
#         """."""
#         outcome = self.state.outcome()
#         delta = 1 if outcome.winner is True else 0
#         return delta
