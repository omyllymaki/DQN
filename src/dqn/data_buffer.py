from collections import deque
from typing import List

import numpy as np

from src.dqn.transition import Transition

def hash(state):
    x = state[0][0].item()
    y = state[0][1].item()
    return (int(x),int(y))

class DataBuffer(object):

    def __init__(self, capacity: int) -> None:
        self.memory = deque([], maxlen=capacity)
        self.rewards = deque([], maxlen=capacity)
        self.state_hashes = deque([], maxlen=capacity)
        self.reward_counts = {}
        self.hashed_state_counts = {}

    def push(self, *args) -> None:
        transition = Transition(*args)
        self.memory.append(transition)

        reward = np.round(transition.reward.item(), 4)
        if self.reward_counts.get(reward) is None:
            self.reward_counts[reward] = 0
        self.reward_counts[reward] += 1
        self.rewards.append(reward)

        state_hash = hash(transition.state)
        if self.hashed_state_counts.get(state_hash) is None:
            self.hashed_state_counts[state_hash] = 0
        self.hashed_state_counts[state_hash] += 1
        self.state_hashes.append(state_hash)

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
