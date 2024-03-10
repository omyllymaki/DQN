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
        """
        Calculate sample priority. Priority needs be within [0, 1], 1 meaning the highest priority.
        """
        raise NotImplementedError


class SigmoidSamplePriority(SamplePriority):

    def __init__(self, k, alpha):
        self.k = k
        self.alpha = alpha

    def apply(self, current_value: float, temporal_diff_error: float) -> float:
        new_value = 2 * (sigmoid(abs(temporal_diff_error), self.k) - 0.5)
        return (1 - self.alpha) * current_value + self.alpha * new_value


class PolynomialSamplePriority(SamplePriority):

    def __init__(self, max_tde_error, beta, alpha):
        self.max_tde_error = max_tde_error
        self.beta = beta
        self.alpha = alpha

    def apply(self, current_value: float, temporal_diff_error: float) -> float:
        new_value = (abs(temporal_diff_error) / self.max_tde_error) ** self.beta
        new_value = min(1, new_value)
        return (1 - self.alpha) * current_value + self.alpha * new_value
