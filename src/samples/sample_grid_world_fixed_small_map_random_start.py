import logging
import time

import numpy as np
import torch
from matplotlib import pyplot as plt

from src.custom_environments.grid_world.grid_world_env import GridWorldEnv
from src.dqn.count_based_exploration import CountBasedExploration
from src.dqn.state_hashing import StateHashing
from src.dqn.memory import Memory
from src.dqn.dqn_agent import DQNAgent
from src.dqn.parameters import Parameters, TrainParameters
from src.dqn.progress_callback import ProgressCallback
from src.dqn.scheduler import LinearScheduler, ConstValueScheduler
from src.dqn.utils import running_mean
from src.samples.sample_utils import ProgressCallbackGridWorld

logging.basicConfig(level=logging.INFO)

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


def main():
    env = GridWorldEnv(size=GRID_SIZE,
                       n_obstacles=15,
                       fixed_target_point=TARGET,
                       fixed_obstacles=OBSTACLES)

    n_actions = env.action_space.n
    state, _ = env.reset()
    n_observations = len(state)

    param = Parameters()
    param.obs_dim = n_observations
    param.action_dim = n_actions
    param.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {param.device} as device")

    train_param = TrainParameters()
    train_param.n_episodes = 1000
    train_param.max_steps_per_episode = 200
    train_param.target_network_update_rate = 0.01
    train_param.progress_cb = ProgressCallbackGridWorld(grid_size=GRID_SIZE,
                                                        target=TARGET,
                                                        obstacles=OBSTACLES,
                                                        vis_period=10,
                                                        n_episodes_to_show=10)
    train_param.eps_scheduler = LinearScheduler(slope=-1 / 700, start_value=1.0, min_value=0)

    hashing = StateHashingXY()
    bonus_reward_coefficient_scheduler = ConstValueScheduler(0.05)
    train_param.count_based_exploration = CountBasedExploration(hashing,
                                                                bonus_reward_coefficient_scheduler)

    agent = DQNAgent(param)

    t1 = time.time()
    agent.train(env, train_param)
    t2 = time.time()
    duration = t2 - t1
    n_step_total = agent.stage.n_steps_total
    print(f"Training took {duration} s for {n_step_total} steps, {n_step_total / duration:0.0f} steps/s")

    plt.show()


if __name__ == "__main__":
    main()
