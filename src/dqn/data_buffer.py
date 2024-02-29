import random
from collections import deque
from typing import List

from src.dqn.transition import Transition


class DataBuffer(object):

    def __init__(self, capacity: int) -> None:
        self.memory = deque([], maxlen=capacity)

    def push(self, *args) -> None:
        self.memory.append(Transition(*args))

    def get_all(self) -> List[Transition]:
        return list(self.memory)

    def get_all_flat(self) -> Transition:
        return Transition(*zip(*self.memory))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.memory, batch_size)

    def get_batch(self, batch_size: int) -> Transition:
        # This converts batch-array of Transitions to Transition of batch-arrays.
        # batch now has keys ('state', 'action', 'next_state', 'reward'), and every key holds len BATCH_SIZE tuple with tensors as elements
        # E.g. batch.state[0] is state of one sample
        sample = self.sample(batch_size)
        return Transition(*zip(*sample))

    def __str__(self):
        first_element = self.memory[0]
        last_element = self.memory[-1]
        s = f"Data buffer size {len(self)}, values: \n {first_element} \n ... \n {last_element}"
        return s

    def __repr__(self):
        return self.__str__()

    def __len__(self) -> int:
        return len(self.memory)

    def __getitem__(self, index):
        return self.memory[index]