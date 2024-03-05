import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch


class Hashing(ABC):
    """
    Abstract interface for hashing.
    """

    @abstractmethod
    def apply(self, *args) -> Any:
        """
        Calculate hash of the inputs
        """
        raise NotImplementedError


class RewardHashing(Hashing):
    """
    Hash reward.
    """

    def __init__(self, resolution: float) -> None:
        self.resolution = resolution

    def apply(self, reward: torch.Tensor) -> int:
        return np.round(reward.item() / self.resolution).astype(int)
