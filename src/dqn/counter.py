import logging
from abc import ABC, abstractmethod
from typing import List, Union

from src.dqn.state_hashing import StateHashing

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


class HashBasedStateCounter(Counter):

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
