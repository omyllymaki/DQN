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
    def apply(self, data_buffer: Memory) -> Optional[Transition]:
        """
        Get sample batch from replay memory
        """
        raise NotImplementedError

    @property
    def batch_size(self):
        return self._batch_size


class RandomSamplingStrategy(SamplingStrategy):

    def __init__(self, batch_size: int) -> None:
        self._batch_size = batch_size

    def apply(self, data_buffer: Memory) -> Optional[Transition]:
        if len(data_buffer) < self.batch_size:
            return None
        sample = random.sample(data_buffer.memory, self.batch_size)
        return Transition(*zip(*sample))


class RewardFreqBasedSamplingStrategy(SamplingStrategy):

    def __init__(self, batch_size: int, f_weighting: Optional[Callable] = None) -> None:
        self._batch_size = batch_size
        if f_weighting is None:
            self.f_weighting = lambda x: 1 / np.sqrt(x)

    def apply(self, data_buffer: Memory) -> Optional[Transition]:
        if len(data_buffer) < self.batch_size:
            return None

        weights = [self.f_weighting(i.reward_count) for i in data_buffer.memory]
        sample = random.choices(population=data_buffer.memory, weights=weights, k=self.batch_size)
        return Transition(*zip(*sample))
