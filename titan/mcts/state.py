"""Generic State Representation."""
from abc import ABC, abstractmethod
from typing import Any

import torch


class State(ABC):
    @abstractmethod
    def is_terminal(self) -> bool:
        """Returns a boolean indicating whether the current state is terminal."""
        return NotImplemented

    @abstractmethod
    def update(self) -> None:
        """Updates the internal state object."""
        return NotImplemented

    @abstractmethod
    def eval(self) -> Any:
        """Evaluates the outcome of the some state and returns some reward."""
        return NotImplemented

    @abstractmethod
    def get_legal_actions(self) -> list:
        """Returns a list of possible future steps from the current state."""
        return NotImplemented

    @abstractmethod
    def get_observation(self) -> torch.Tensor:
        """Returns the observation tensor."""
        return NotImplemented
