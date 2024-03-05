from collections import deque
from typing import List, Callable, Optional

import numpy as np

from src.dqn.transition import Transition


class Memory(object):

    def __init__(self,
                 capacity: int,
                 reward_hashing: Optional[Callable] = None,
                 state_hashing: Optional[Callable] = None) -> None:
        self.memory = deque([], maxlen=capacity)
        self.reward_hashing = reward_hashing
        self.state_hashing = state_hashing
        self.hashed_reward_counts = {}
        self.hashed_state_counts = {}

    def push(self, state, action, next_state, reward) -> None:
        reward_count = None
        if self.reward_hashing is not None:
            reward_hash = self.reward_hashing.apply(reward)
            if self.hashed_reward_counts.get(reward_hash) is None:
                self.hashed_reward_counts[reward_hash] = 0
            self.hashed_reward_counts[reward_hash] += 1
            reward_count = self.hashed_reward_counts[reward_hash]

        state_count = None
        if self.state_hashing is not None:
            state_hash = self.state_hashing.apply(state)
            if self.hashed_state_counts.get(state_hash) is None:
                self.hashed_state_counts[state_hash] = 0
            self.hashed_state_counts[state_hash] += 1
            state_count = self.hashed_state_counts[state_hash]

        transition = Transition(state=state,
                                action=action,
                                next_state=next_state,
                                reward=reward,
                                reward_count=reward_count,
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
