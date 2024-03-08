import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import List, Union

import torch

from src.dqn.state_hashing import StateHashing, ModelStateHashing

logger = logging.getLogger(__name__)


class Counter(ABC):
    """
    Abstract interface for counter.
    """

    @abstractmethod
    def push(self, *args) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_counts(self, *args) -> Union[int, List[int]]:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError


class SimpleHashedStateCounter(Counter):
    """
    Hashed state counter. Uses fixed hashing that doesn't depend on the data.
    """

    def __init__(self, hashing: StateHashing):
        self.hashing = hashing
        self.counts = {}

    def push(self, state) -> None:
        n1 = len(self.counts)
        state_hash = self.hashing.hash(state)[0]
        if self.counts.get(state_hash) is None:
            self.counts[state_hash] = 0
        self.counts[state_hash] += 1
        n2 = len(self.counts)
        if n2 > n1:
            logger.debug(f"Number of unique states in counter {n2}")

    def get_counts(self, states):
        state_hashes = self.hashing.hash(states)
        return [self.counts[h] for h in state_hashes]

    def reset(self) -> None:
        self.counts = {}


class ModelHashedStateCounter(Counter):
    """
    Hashed state counter. Uses hashing that is fitted with the collected data.
    """

    def __init__(self, hashing: ModelStateHashing, buffer_size=30000, fit_period=10000):
        self.hashing = hashing
        self.counter = 0
        self.counts = {}
        self.buffer = deque([], maxlen=buffer_size)
        self.fit_period = fit_period
        self.first_fit_done = False

    def push(self, state) -> None:
        self.buffer.append(state)
        self.counter += 1

        if self.counter % self.fit_period == 0:
            stacked_buffer = torch.cat(list(self.buffer), dim=0)
            state_hashes = self.hashing.fit(stacked_buffer)
            self.counts = {}
            for h in state_hashes:
                if self.counts.get(h) is None:
                    self.counts[h] = 0
                self.counts[h] += 1
            self.counter = 0
            self.first_fit_done = True

    def get_counts(self, states):
        if self.first_fit_done:
            state_hashes = self.hashing.hash(states)
            return [self.counts.get(h, 1) for h in state_hashes]
        else:
            return [1 for _ in range(states.shape[0])]

    def reset(self) -> None:
        self.counts = {}
