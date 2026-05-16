import numpy as np
from abc import ABC, abstractmethod


class Policy(ABC):

    @abstractmethod
    def mount(self) -> None:
        pass

    @abstractmethod
    def act(self, s: np.ndarray) -> int:
        pass
