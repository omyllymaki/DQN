import math
from abc import ABC, abstractmethod


class Scheduler(ABC):
    """
    Abstract interface for scheduler.
    Scheduler changes value of variable during training using determined rules.
    """

    @abstractmethod
    def apply(self, i_episode, n_steps_episode, n_steps_total) -> float:
        """
        Calculate new value of variable.
        """
        raise NotImplementedError


class ExpDecayScheduler(Scheduler):

    def __init__(self, start=0.9, end=0.05, decay=10000):
        self.start = start
        self.end = end
        self.decay = decay

    def apply(self, i_episode, n_steps_episode, n_steps_total) -> float:
        return self.end + (self.start - self.end) * math.exp(-1. * n_steps_total / self.decay)

    def __str__(self):
        return f"exponential decay scheduler: start {self.start}, end {self.end}, decay {self.decay}"

    def __repr__(self):
        return self.__str__()


class ConstValueScheduler(Scheduler):

    def __init__(self, value):
        self.value = value

    def apply(self, i_episode, n_steps_episode, n_steps_total) -> float:
        return self.value

    def __str__(self):
        return f"const value scheduler: value {self.value}"

    def __repr__(self):
        return self.__str__()


class PowerScheduler(Scheduler):

    def __init__(self, coefficient=0.9999, min_value=0.05):
        self.coefficient = coefficient
        self.min_value = min_value

    def apply(self, i_episode, n_steps_episode, n_steps_total) -> float:
        return max(self.min_value, self.coefficient ** n_steps_total)

    def __str__(self):
        return f"power scheduler: coefficient {self.coefficient}, min value {self.min_value}"

    def __repr__(self):
        return self.__str__()


class LinearScheduler(Scheduler):

    def __init__(self, slope, start_value=1.0, min_value=0):
        self.slope = slope
        self.start_value = start_value
        self.min_value = min_value

    def apply(self, i_episode, n_steps_episode, n_steps_total) -> float:
        return max(self.start_value + i_episode * self.slope, self.min_value)

    def __str__(self):
        return f"linear scheduler: n episodes in total {self.n_episodes_in_total}, min value {self.min_value}, max value {self.max_value}"

    def __repr__(self):
        return self.__str__()
