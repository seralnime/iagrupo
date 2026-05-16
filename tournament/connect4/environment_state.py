# Abstract
from abc import ABC, abstractmethod

# Types
from typing import Any


class EnvironmentState(ABC):
    """
    Abstract base class representing the state of a reinforcement learning environment.
    """

    @abstractmethod
    def is_final(self) -> bool:
        """
        Determines whether the current state is a final (terminal) state.

        Returns
        -------
        bool
            True if the game has reached a final state (win or draw); False otherwise.
        """
        pass

    @abstractmethod
    def is_applicable(self, event: Any) -> bool:
        """
        Checks whether an event (e.g., an action) is applicable in the current state.

        Parameters
        ----------
        event : Any
            The event to test (e.g., column index or action descriptor).

        Returns
        -------
        bool
            True if the event can occur in the current state; False otherwise.
        """
        pass

    @abstractmethod
    def transition(self, event: Any) -> "EnvironmentState":
        """
        Places a tile in the specified column for the active player.

        Parameters
        ----------
        event: Any
            The event that occurs

        Returns
        -------
        EnvironmentState
            New environment state after the move.

        Raises
        ------
        ValueError
            If the move is invalid (e.g., column is full or game is over).
        """
        pass
