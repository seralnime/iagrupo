# Abstract
from connect4.environment_state import EnvironmentState

# Types
from typing import Any

# Libraries
import numpy as np
import matplotlib.pyplot as plt


class ConnectState(EnvironmentState):
    ROWS = 6
    COLS = 7

    def __init__(self, board: np.ndarray | None = None, player: int = -1):
        if board is None:
            self.board = np.zeros((self.ROWS, self.COLS), dtype=int)
        else:
            self.board = board.copy()
        self.player = player  # -1 = Red, 1 = Yellow type: ignore

    def is_final(self) -> bool:
        return self.get_winner() != 0 or not any(self.board[0] == 0)

    def is_applicable(self, event: Any) -> bool:
        return (
            isinstance(event, int)
            and 0 <= event < self.COLS
            and self.is_col_free(event)
            and not self.is_final()
        )

    def get_winner(self) -> int:
        # Check all 4 directions
        for r in range(self.ROWS):
            for c in range(self.COLS):
                player = self.board[r, c]
                if player == 0:
                    continue

                # Right
                if c + 3 < self.COLS and all(
                    self.board[r, c + i] == player for i in range(4)
                ):
                    return player
                # Down
                if r + 3 < self.ROWS and all(
                    self.board[r + i, c] == player for i in range(4)
                ):
                    return player
                # Diagonal right-down
                if (
                    r + 3 < self.ROWS
                    and c + 3 < self.COLS
                    and all(self.board[r + i, c + i] == player for i in range(4))
                ):
                    return player
                # Diagonal left-down
                if (
                    r + 3 < self.ROWS
                    and c - 3 >= 0
                    and all(self.board[r + i, c - i] == player for i in range(4))
                ):
                    return player

        return 0

    def is_col_free(self, col: int) -> bool:
        return self.board[0, col] == 0

    def get_heights(self) -> list[int]:
        heights = []
        for c in range(self.COLS):
            col = self.board[:, c]
            for r in range(self.ROWS):
                if col[r] != 0:
                    heights.append(self.ROWS - r)
                    break
            else:
                heights.append(0)
        return heights

    def get_free_cols(self) -> list[int]:
        return [c for c in range(self.COLS) if self.is_col_free(c)]

    def transition(self, col: int) -> "ConnectState":
        if not self.is_applicable(col):
            raise ValueError(f"Move not allowed in column {col}.")

        new_board = self.board.copy()
        for r in reversed(range(self.ROWS)):
            if new_board[r, col] == 0:
                new_board[r, col] = self.player
                break

        return ConnectState(new_board, -self.player)

    def show(self, size: int = 1500, ax: plt.Axes | None = None) -> None:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None

        pos_red = np.where(self.board == -1)
        pos_yellow = np.where(self.board == 1)

        ax.scatter(pos_yellow[1] + 0.5, 5.5 - pos_yellow[0], color="yellow", s=size)
        ax.scatter(pos_red[1] + 0.5, 5.5 - pos_red[0], color="red", s=size)

        ax.set_ylim([0, self.board.shape[0]])
        ax.set_xlim([0, self.board.shape[1]])
        ax.set_xticks(np.arange(self.board.shape[1] + 1))
        ax.set_yticks(np.arange(self.board.shape[0] + 1))
        ax.grid(True)

        ax.set_title("Connect Four")

        if fig is not None:
            plt.show()
