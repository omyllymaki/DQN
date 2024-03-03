import time
import unittest

import gymnasium as gym
import numpy as np

from src.dqn.dqn_agent import DQNAgent
from src.dqn.parameters import Parameters, TrainParameters
from src.dqn.progress_callback import ProgressCallbackVisDuration
from src.dqn.scheduler import ExpDecayScheduler


class CartPoleTests(unittest.TestCase):

    def setUp(self):
        self.env = gym.make("CartPole-v1")

    def test_training_with_default_parameters(self):
        param, train_param = self._init_param()

        agent = DQNAgent(param)

        t1 = time.time()
        results = agent.train(self.env, train_param)
        t2 = time.time()
        duration = t2 - t1
        n_step_total = agent.steps_done_total
        n_steps_per_second = n_step_total / duration
        print(f"Training took {duration} s for {n_step_total} steps, {n_steps_per_second:0.0f} steps/s")

        last_episodes = results[-50:]
        avg_duration_last_episodes = np.mean([len(item) for item in last_episodes])
        print(f"Average duration in last episodes: {avg_duration_last_episodes}")

        self.assertGreater(avg_duration_last_episodes, 150)
        self.assertGreater(n_steps_per_second, 150)

    def _init_param(self):
        n_actions = self.env.action_space.n
        state, _ = self.env.reset()
        n_observations = len(state)

        param = Parameters()
        param.obs_dim = n_observations
        param.action_dim = n_actions

        train_param = TrainParameters()
        train_param.eps_scheduler = ExpDecayScheduler(start=0.9, end=0.05, decay=1000)
        train_param.n_episodes = 250
        train_param.max_steps_per_episode = 500
        train_param.progress_cb = ProgressCallbackVisDuration(50)

        return param, train_param


if __name__ == '__main__':
    unittest.main()
