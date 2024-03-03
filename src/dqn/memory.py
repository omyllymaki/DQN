from collections import deque
from typing import List, Callable, Optional

import numpy as np

from src.dqn.transition import Transition


class Memory(object):

    def __init__(self, capacity: int, state_hash_func: Optional[Callable] = None) -> None:
        self.memory = deque([], maxlen=capacity)
        self.state_hash_func = state_hash_func
        self.reward_counts = {}
        self.hashed_state_counts = {}

    def push(self, state, action, next_state, reward) -> None:
        reward_item = np.round(reward.item(), 4)
        if self.reward_counts.get(reward_item) is None:
            self.reward_counts[reward_item] = 0
        self.reward_counts[reward_item] += 1

        state_count = None
        if self.state_hash_func is not None:
            state_hash = self.state_hash_func(state)
            if self.hashed_state_counts.get(state_hash) is None:
                self.hashed_state_counts[state_hash] = 0
            self.hashed_state_counts[state_hash] += 1
            state_count = self.hashed_state_counts[state_hash]

        transition = Transition(state=state,
                                action=action,
                                next_state=next_state,
                                reward=reward,
                                reward_count=self.reward_counts[reward_item],
                                state_count=state_count)
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
