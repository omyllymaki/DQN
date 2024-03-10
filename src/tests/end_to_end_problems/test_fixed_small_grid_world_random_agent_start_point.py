import time
import unittest

import numpy as np
import torch

from src.custom_environments.grid_world.grid_world_env import GridWorldEnv
from src.dqn.count_based_exploration import CountBasedExploration
from src.dqn.counter import SimpleHashedStateCounter
from src.dqn.dqn_agent import DQNAgent
from src.dqn.sample_priority import SigmoidSamplePriority, PolynomialSamplePriority
from src.dqn.sampling_strategy import PrioritizedSamplingStrategy
from src.dqn.state_hashing import StateHashing
from src.dqn.parameters import Parameters, TrainParameters
from src.dqn.progress_callback import ProgressCallbackVisSumReward
from src.dqn.scheduler import LinearScheduler, ConstValueScheduler

GRID_SIZE = 30
TARGET = (15, 15)
OBSTACLES = (
    (0, 15),
    (10, 9),
    (20, 24),
    (28, 27),
    (2, 27),
    (3, 4),
    (1, 26),
    (1, 13),
    (26, 13),
    (27, 0),
    (16, 8),
    (5, 1),
    (28, 29),
    (26, 26),
    (27, 0)
)


class StateHashingXY(StateHashing):
    def hash(self, states):
        agent_xy = states[:, [0, 1]]
        hashes = torch.round(agent_xy).to(torch.int)
        return [(h[0].item(), h[1].item()) for h in hashes]


class FixedSmallGridWorldRandomAgentStartPointTests(unittest.TestCase):

    def setUp(self):
        self.env = GridWorldEnv(size=GRID_SIZE,
                                fixed_target_point=TARGET,
                                fixed_obstacles=OBSTACLES,
                                target_reward=10,
                                obstacle_reward=-10,
                                default_reward=-0.05)

    def test_training_with_default_parameters(self):
        param, train_param = self._init_param()

        agent = DQNAgent(param)

        t1 = time.time()
        results = agent.train(self.env, train_param)
        t2 = time.time()
        duration = t2 - t1
        n_step_total = agent.stage.n_steps_total
        n_steps_per_second = n_step_total / duration
        print(f"Training took {duration} s for {n_step_total} steps, {n_steps_per_second:0.0f} steps/s")

        last_episodes = results[-50:]
        avg_cum_reward_last_episodes = np.mean([np.sum(item) for item in last_episodes])
        print(f"Average cumulative rewards in last episodes: {avg_cum_reward_last_episodes}")

        self.assertGreater(avg_cum_reward_last_episodes, 3)
        self.assertGreater(n_steps_per_second, 150)

    def test_training_with_count_based_exploration(self):
        param, train_param = self._init_param()

        exploration_bonus_reward_coeff_scheduler = ConstValueScheduler(0.05)
        state_hashing = StateHashingXY()
        counter = SimpleHashedStateCounter(state_hashing)
        train_param.count_based_exploration = CountBasedExploration(counter,
                                                                    exploration_bonus_reward_coeff_scheduler)

        agent = DQNAgent(param)

        t1 = time.time()
        results = agent.train(self.env, train_param)
        t2 = time.time()
        duration = t2 - t1
        n_step_total = agent.stage.n_steps_total
        n_steps_per_second = n_step_total / duration
        print(f"Training took {duration} s for {n_step_total} steps, {n_steps_per_second:0.0f} steps/s")

        last_episodes = results[-50:]
        avg_cum_reward_last_episodes = np.mean([np.sum(item) for item in last_episodes])
        print(f"Average cumulative rewards in last episodes: {avg_cum_reward_last_episodes}")

        self.assertGreater(avg_cum_reward_last_episodes, 3)
        self.assertGreater(n_steps_per_second, 70)

    def test_priority_sampling(self):
        param, train_param = self._init_param()
        train_param.sampling_strategy = PrioritizedSamplingStrategy(128)
        train_param.sample_priory_update = PolynomialSamplePriority(max_tde_error=10, beta=1, alpha=0.5)

        agent = DQNAgent(param)

        t1 = time.time()
        results = agent.train(self.env, train_param)
        t2 = time.time()
        duration = t2 - t1
        n_step_total = agent.stage.n_steps_total
        n_steps_per_second = n_step_total / duration
        print(f"Training took {duration} s for {n_step_total} steps, {n_steps_per_second:0.0f} steps/s")

        last_episodes = results[-50:]
        avg_cum_reward_last_episodes = np.mean([np.sum(item) for item in last_episodes])
        print(f"Average cumulative rewards in last episodes: {avg_cum_reward_last_episodes}")

        self.assertGreater(avg_cum_reward_last_episodes, 3)
        self.assertGreater(n_steps_per_second, 50)

    def _init_param(self):
        n_actions = self.env.action_space.n
        state, _ = self.env.reset()
        n_observations = len(state)

        param = Parameters()
        param.obs_dim = n_observations
        param.action_dim = n_actions

        train_param = TrainParameters()
        train_param.n_episodes = 1000
        train_param.max_steps_per_episode = 200
        train_param.eps_scheduler = LinearScheduler(slope=-1 / 700, start_value=1.0, min_value=0)
        train_param.progress_cb = ProgressCallbackVisSumReward(50)

        return param, train_param


if __name__ == '__main__':
    unittest.main()
