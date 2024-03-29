from abc import ABC, abstractmethod

import numpy as np
from matplotlib import pyplot as plt

from src.dqn.memory import Memory
from src.dqn.progress_data import ProgressData
from src.dqn.utils import running_mean


class ProgressCallback(ABC):
    """
    Abstract interface for progress callbacks.
    """

    @abstractmethod
    def push(self, data: ProgressData) -> None:
        """
        Push data collected during episode.

        Args:
            data (ProgressData): data from the episode.
        """
        raise NotImplementedError

    @abstractmethod
    def apply(self) -> None:
        """
        Apply defined steps and rendering to data.
        """
        raise NotImplementedError


class ProgressCallbackSimple(ProgressCallback):

    def __init__(self):
        self.data = None

    def push(self, data: ProgressData) -> None:
        self.data = data

    def apply(self) -> None:
        print(self.data)


class ProgressCallbackVisLatestRewards(ProgressCallback):

    def __init__(self, running_mean_window_size=50) -> None:
        super().__init__()
        self.running_mean_window_size = running_mean_window_size
        self.rewards = []

    def push(self, data: ProgressData) -> None:
        last_transition = data.transitions[-1]
        self.rewards.append(last_transition.reward.cpu().numpy())

    def apply(self) -> None:
        plt.figure(1)
        plt.cla()
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.plot(self.rewards, "b-")
        if self.running_mean_window_size is not None:
            plt.plot(running_mean(self.rewards, self.running_mean_window_size), "r-", linewidth=2)
        plt.plot()
        plt.pause(0.001)


class ProgressCallbackVisDuration(ProgressCallback):

    def __init__(self, running_mean_window_size=50) -> None:
        super().__init__()
        self.running_mean_window_size = running_mean_window_size
        self.durations = []

    def push(self, data: ProgressData) -> None:
        duration = len(data.transitions)
        self.durations.append(duration)

    def apply(self) -> None:
        plt.figure(1)
        plt.cla()
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(self.durations, "b-")
        if self.running_mean_window_size is not None:
            plt.plot(running_mean(self.durations, self.running_mean_window_size), "r-", linewidth=2)
        plt.plot()
        plt.pause(0.001)


class ProgressCallbackVisSumReward(ProgressCallback):

    def __init__(self, running_mean_window_size=50) -> None:
        super().__init__()
        self.running_mean_window_size = running_mean_window_size
        self.rewards = []

    def push(self, data: ProgressData) -> None:
        episode_rewards_sum = np.sum([r.reward.item() for r in data.transitions])
        self.rewards.append(episode_rewards_sum)

    def apply(self) -> None:
        plt.figure(1)
        plt.cla()
        plt.xlabel('Episode')
        plt.ylabel('Cumulative reward')
        plt.plot(self.rewards, "b-")
        if self.running_mean_window_size is not None:
            plt.plot(running_mean(self.rewards, self.running_mean_window_size), "r-", linewidth=2)
        plt.plot()
        plt.pause(0.001)
