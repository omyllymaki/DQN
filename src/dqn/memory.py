from collections import deque
from typing import List, Callable, Optional

import numpy as np

from src.dqn.transition import Transition


class Memory(object):

    def __init__(self,capacity: int) -> None:
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, next_state, reward) -> None:
        transition = Transition(state=state,
                                action=action,
                                next_state=next_state,
                                reward=reward,
                                priority=1)
        self.memory.append(transition)

    def get_all(self) -> List[Transition]:
        return list(self.memory)

    def get_all_flat(self) -> Transition:
        return Transition(*zip(*self.memory))

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
