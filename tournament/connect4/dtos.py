from pydantic import BaseModel, ConfigDict, Field
from connect4.policy import Policy
import numpy as np

State = np.ndarray
Action = int
Participant = tuple[str, Policy]
Versus = list[tuple[Participant | None, Participant | None]]


class Game(list[tuple[State, Action]]):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class Match(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    player_a: str = Field(description="First Player")
    player_b: str = Field(description="Second Player")

    player_a_wins: int = Field(default=0, description="Games won by First Player.")
    player_b_wins: int = Field(default=0, description="Games won by Second Player.")
    draws: int = Field(default=0, description="Games ended in draw.")

    games: list[Game] = Field(
        default=[],
        description="List of the history of each game, a state-action pair list produced by the alternating sequence of player actions.",
    )
