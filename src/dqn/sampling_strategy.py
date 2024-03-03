import random
from abc import ABC, abstractmethod
from typing import Optional, Callable

import numpy as np

from src.dqn.data_buffer import DataBuffer
from src.dqn.transition import Transition


class SamplingStrategy(ABC):
    _batch_size = 0
    """
    Abstract interface for sampling strategy.
    SamplingStrategy determines how sample batch is drawn from the replay memory.
    """

    @abstractmethod
    def apply(self, data_buffer: DataBuffer) -> Optional[Transition]:
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

    def apply(self, data_buffer: DataBuffer) -> Optional[Transition]:
        if len(data_buffer) < self.batch_size:
            return None, None
        indices = np.arange(0, len(data_buffer.memory)).tolist()
        sample_indices = random.sample(indices, self.batch_size)
        sample = [data_buffer.memory[index] for index in sample_indices]
        sample_state_hashes = [data_buffer.state_hashes[index] for index in sample_indices]
        state_hash_counts = [data_buffer.hashed_state_counts[h] for h in sample_state_hashes]
        return Transition(*zip(*sample)), state_hash_counts


class RewardFreqBasedSamplingStrategy(SamplingStrategy):

    def __init__(self, batch_size: int, f_weighting: Optional[Callable] = None) -> None:
        self._batch_size = batch_size
        if f_weighting is None:
            self.f_weighting = lambda x: 1 / np.sqrt(x)

    def apply(self, data_buffer: DataBuffer) -> Optional[Transition]:
        if len(data_buffer) < self.batch_size:
            return None

        reward_weights = {}
        weight_sum = 0
        for reward_value, count in data_buffer.reward_counts.items():
            weight = self.f_weighting(count)
            reward_weights[reward_value] = weight
            weight_sum += weight

        for k, w in reward_weights.items():
            reward_weights[k] = w / weight_sum

        sample_weights = [reward_weights[r] for r in data_buffer.rewards]
        sample = random.choices(population=data_buffer.memory, weights=sample_weights, k=self.batch_size)
        return Transition(*zip(*sample))
