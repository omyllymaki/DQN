from dataclasses import dataclass

import torch
from torch import Tensor

from src.dqn.counter import SimpleHashedStateCounter, Counter
from src.dqn.state_hashing import StateHashing
from src.dqn.scheduler import Scheduler, Stage


class CountBasedExploration:
    """
    This class implements exploration bonus reward based on hashed state count.
    """

    def __init__(self,
                 counter: Counter,
                 bonus_reward_coefficient_scheduler: Scheduler
                 ):
        self.counter = counter
        self.bonus_reward_coefficient_scheduler = bonus_reward_coefficient_scheduler

    def push(self, state: Tensor):
        self.counter.push(state)

    def get_bonus_rewards(self, state_batch: torch.tensor, stage: Stage):
        coeff = self.bonus_reward_coefficient_scheduler.apply(stage)
        state_counts = self.counter.get_counts(state_batch)
        bonus_reward = coeff / torch.sqrt(torch.Tensor(state_counts))
        return bonus_reward
