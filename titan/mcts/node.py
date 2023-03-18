import numpy as np


class Node:
    def __init__(self, state, parent=None):
        self.n_k, self.w_k = 0, 0
        self.parent = parent
        self.state = state
        childs = set()
    
    def select(self, set_of_actions):
        """First step of MCTS."""
        pass

    def expand(self, state):
        """Expands the tree."""
        new_node = Node(state, self)
        print(new_node)
        childs.add(new_node)
        
    def backpropagte(self) -> None:
        pass

    def simulate(self):
        pass

    def ucb(self, t, n) -> float:
        return t + 2 * np.sqrt((np.log(self.n_k) / n))
    
