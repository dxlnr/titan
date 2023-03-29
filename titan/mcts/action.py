"""Action State History"""


class ActionState:
    """Action state container used to keep track of the actions executed."""

    def __init__(self):
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.to_play_history = []
        self.child_visits = []
        self.root_values = []

        # # Timesteps, history and current
        # self.T, self.t = 8, 0
        # # Representation of the board inputs which gets feeded to h (representation).
        # self.enc_state = torch.zeros([self.N, self.N, (self.M * self.T + self.L)])

    def add_action(self, action):
        """Appends an action."""
        self.history.append(action)

    def last_action(self):
        """Returns the last action."""
        pass

    def action_space(self) -> list:
        """Returns a list of all actions performed."""
        pass

    def to_play(self):
        pass
