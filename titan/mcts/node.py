# from typing import Self

import chess
import math
from titan.mcts.state import State


class Node:
    def __init__(self, move: str = '', parent = None):
        self.n_k, self.w_k = 0, 0
        self.move = move
        self.parent = parent
        # self.state = state
        self.children = list()
    
    def __repr__(self):
        return f"Node( nk: {self.n_k}, w_k: {self.w_k}, move: {self.move}, parent: {self.parent}."

    # def select(self):
    #     if not terminal(state):
    #         for each non-isomorphic legal move m of state:
    #     nc = Node(m, self) # new child node
    #     self.children.append(nc)
    #     while 
        
    # def select(self):
    #     """First step of MCTS."""
    #     if len(self.children) == len(self.state.get_possible_moves()):
    #         idx = self.children.index(min(self.children, key=lambda x: x.ucb(x.w_k, x.n_k)))
    #         return self.children[idx]
    #     else:
    #         move = self.state.get_possible_moves()[len(self.children)] 

    #         new_board = self.state.obj.copy()
    #         new_board.push(chess.Move.from_uci(str(move)))
            
    #         new_node = Node(State(new_board), self)
    #         self.children.append(new_node)
    #         return new_node

    def expand(self, state):
        """Expands the tree."""
        if not state.is_terminated():
            for m in state.get_possible_moves():
                c_node = Node(m, self)
                self.children.append(c_node)

    def is_leaf_node(self) -> bool:
        """."""
        return len(self.children) == 0

    def is_root_node(self) -> bool:
        """."""
        return self.parent is None

    def propagate(self, delta: int) -> None:
        """Update nk & wk values accordingly."""
        self.n_k = self.n_k + 1
        self.w_k = self.w_k + delta

    def ucb(self) -> float:
        """."""
        # print("self.wk, self.nk, parent.nk", self.w_k, self.n_k, self.parent.n_k)
        return (self.w_k/self.n_k) + 2 * math.sqrt((math.log(self.parent.n_k) / self.n_k))


    
