"""Action State History"""


class ActionHistory:
    """Action state container used to keep track of the actions executed."""

    def __init__(self):
        self.observation_history = []
        self.action_history = []
        self.reward_history = []
        self.to_play_history = []
        self.child_visits = []
        self.root_values = []

    def add_action(self, action: int) -> None:
        """Appends an action."""
        self.action_history.append(action)

    def last_action(self):
        """Returns the last action."""
        return self.action_history[-1]

    def action_space(self) -> list:
        """Returns a list of all actions performed."""
        pass

    def to_play(self):
        pass
