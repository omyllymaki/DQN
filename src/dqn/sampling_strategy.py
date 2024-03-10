import random
from abc import ABC, abstractmethod
from typing import Optional, Callable, Tuple, List

import numpy as np

from src.dqn.memory import Memory
from src.dqn.transition import Transition


class SamplingStrategy(ABC):
    _batch_size = 0
    """
    Abstract interface for sampling strategy.
    SamplingStrategy determines how sample batch is drawn from the replay memory.
    """

    @abstractmethod
    def apply(self, data_buffer: Memory) -> Tuple[Optional[Transition], Optional[List[int]]]:
        """
        Get sample batch from replay memory. Return sample batch and indices of the sample.
        """
        raise NotImplementedError

    @property
    def batch_size(self):
        return self._batch_size


class RandomSamplingStrategy(SamplingStrategy):

    def __init__(self, batch_size: int) -> None:
        self._batch_size = batch_size

    def apply(self, data_buffer: Memory) -> Tuple[Optional[Transition], Optional[List[int]]]:
        if len(data_buffer) < self.batch_size:
            return None, None
        indices = np.arange(len(data_buffer)).tolist()
        sample_indices = random.sample(indices, self.batch_size)
        sample = [data_buffer[i] for i in sample_indices]
        return Transition(*zip(*sample)), sample_indices


class TimeBasedSamplingStrategy(SamplingStrategy):
    """
    Sampling strategy that samples transitions based on time received (index in memory).
    This can be used e.g. to favor more recent transitions in sampling.
    """

    def __init__(self, batch_size: int, f_weighting: Optional[Callable] = None) -> None:
        self._batch_size = batch_size
        if f_weighting is None:
            self.f_weighting = lambda x: x

    def apply(self, data_buffer: Memory) -> Tuple[Optional[Transition], Optional[List[int]]]:
        if len(data_buffer) < self.batch_size:
            return None

        indices = np.arange(len(data_buffer)).tolist()
        weights = [self.f_weighting(i) for i in range(len(data_buffer))]
        sample_indices = random.choices(population=indices, weights=weights, k=self.batch_size)
        sample = [data_buffer[i] for i in sample_indices]
        return Transition(*zip(*sample)), sample_indices


class PrioritizedSamplingStrategy(SamplingStrategy):
    """
    Sampling strategy that samples transitions based on priority.
    """

    def __init__(self, batch_size: int) -> None:
        self._batch_size = batch_size

    def apply(self, data_buffer: Memory) -> Tuple[Optional[Transition], Optional[List[int]]]:
        if len(data_buffer) < self.batch_size:
            return None, None
        indices = np.arange(len(data_buffer)).tolist()
        weights = [data_buffer[i].priority for i in indices]
        sample_indices = random.choices(population=indices, weights=weights, k=self.batch_size)
        sample = [data_buffer[i] for i in sample_indices]
        return Transition(*zip(*sample)), sample_indices
