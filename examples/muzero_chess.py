"""Predict chess moves using MCTS."""
import sys
from pathlib import Path

# Adds root directory to path.
sys.path.append(str(Path(__file__).parent.parent))

from titan.game.chess_state import Chess
from titan.game.chess_board import Board

if __name__ == "__main__":
    # Initiate some board state.
    state = Chess(Board())
    print(state.state.current_board)
    # state.encode_board_state()
    
    print("")
    d1 = state.decode_board_state()
    print(d1)

    action_idxs = []
    for action in state.state.actions():
        print(action)
        if action != []:
            initial_pos,final_pos,underpromote = action
            print(type(initial_pos))
            action_idxs.append(state.encode_action(initial_pos, final_pos, underpromote))
    
    print(action_idxs)
    # acts = state.encode_action()
    # print(acts)
    # last_move = np.argmax(dataset[-1][1])
    # b = state.decode_board_state()
    # act = state.decode_action()
    # print(act)

    # b.move_piece(act[0][0],act[1][0],act[2][0])
    # for i in range(len(dataset)):
    #     board = ed.decode_board(dataset[i][0])
    #     fig = vb(board.current_board)



# import chess

# from titan.mcts import mcts
# from titan.mcts.chess_state import Chess
# # from titan.models.muzero import compute_repr


# if __name__ == "__main__":
#     # Initiate some board state.
#     state = Chess(chess.Board())
#     s = state.encode_epd() 
#     print(s)
#     state.encode_board_state()
    
#     state.update("e2e4")
#     state.update("e7e5")
#     state.update("g1f3")
#     state.update("b8c6")
#     state.update("f1c4")
#     state.update("g8f6")
#     state.update("f3g5")
#     state.update("d7d5")

#     print(state.t)
#     print(state.state.epd())

#     dec1 = state.decode_board_state(timestep=1)
#     dec3 = state.decode_board_state(timestep=3)
#     dec7 = state.decode_board_state(timestep=7)
#     print("")
#     print(dec1)
#     print("")
#     print(dec3)
#     print("")
#     print(dec7)

#     print("")
#     print(state.state.legal_moves)
#     print(list(state.state.legal_moves)[0].from_square)
#     m = list(state.state.legal_moves)[0]
#     print(type(m))
#     print(m)
#     state.encode_board_action(m)
#     # h = compute_repr(state)

#     # print(h)
