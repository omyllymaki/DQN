from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


def sigmoid(x, k):
    return 1 / (1 + np.exp(-1 * (x / k)))


class SamplePriority(ABC):
    """
    Abstract interface to calculate sample priority.
    """

    @abstractmethod
    def apply(self, current_value: float, temporal_diff_error: float) -> float:
        raise NotImplementedError


class SigmoidSamplePriority(ABC):

    def __init__(self, k, alpha):
        self.k = k
        self.alpha = alpha

    def apply(self, current_value: float, temporal_diff_error: float) -> float:
        new_value = 2 * (sigmoid(abs(temporal_diff_error), self.k) - 0.5)
        return (1 - self.alpha) * current_value + self.alpha * new_value
