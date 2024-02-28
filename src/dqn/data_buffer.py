import random
from collections import deque
from typing import List

from src.dqn.transition import Transition


class DataBuffer(object):

    def __init__(self, capacity: int) -> None:
        self.memory = deque([], maxlen=capacity)

    def push(self, *args) -> None:
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.memory, batch_size)

    def get_batch(self, batch_size: int) -> Transition:
        # This converts batch-array of Transitions to Transition of batch-arrays.
        # batch now has keys ('state', 'action', 'next_state', 'reward'), and every key holds len BATCH_SIZE tuple with tensors as elements
        # E.g. batch.state[0] is state of one sample
        sample = self.sample(batch_size)
        return Transition(*zip(*sample))

    def __len__(self) -> int:
        return len(self.memory)
