import logging
import time
from copy import deepcopy

import numpy as np
import torch
from matplotlib import pyplot as plt

from src.custom_environments.grid_world.grid_world_env import GridWorldEnv
from src.dqn.count_based_exploration import CountBasedExploration
from src.dqn.counter import ModelHashedStateCounter, SimpleHashedStateCounter
from src.dqn.dqn_agent import DQNAgent
from src.dqn.parameters import Parameters, TrainParameters
from src.dqn.sample_priority import PolynomialSamplePriority
from src.dqn.sampling_strategy import PrioritizedSamplingStrategy, RandomSamplingStrategy
from src.dqn.scheduler import LinearScheduler, ConstValueScheduler
from src.dqn.state_hashing import PCAStateHashing, StateHashing
from src.dqn.utils import running_mean
from src.samples.sample_utils import ProgressCallbackGridWorld

logging.basicConfig(level=logging.WARNING)

GRID_SIZE = 30
TARGET = (20, 20)
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
BATCH_SIZE = 64
N_ITER_PER_METHOD = 20


def get_train_param():
    train_param = TrainParameters()
    train_param.n_episodes = 1000
    train_param.max_steps_per_episode = 200
    train_param.target_network_update_rate = 0.1
    train_param.eps_scheduler = LinearScheduler(slope=-0.7 / 700, start_value=0.7, min_value=0)
    # train_param.progress_cb = ProgressCallbackGridWorld(GRID_SIZE, TARGET, OBSTACLES, 10, 10)
    train_param.progress_cb = None
    train_param.sampling_strategy = RandomSamplingStrategy(batch_size=BATCH_SIZE)
    train_param.buffer_size = 15000
    return train_param


def calculate_smoothed_cumulative_reward(rewards, window_size=50):
    return running_mean([np.sum(r) for r in rewards], window_size)


def run_test(param, train_param, env, name):
    print(f"Start training: {name}")
    ys = []
    for k in range(N_ITER_PER_METHOD):
        print(f"Iteration {k + 1}/{N_ITER_PER_METHOD}")
        agent = DQNAgent(deepcopy(param))
        t1 = time.time()
        rewards = agent.train(deepcopy(env), deepcopy(train_param))
        t2 = time.time()
        duration = t2 - t1
        n_step_total = agent.stage.n_steps_total
        print(f"Training took {duration} s for {n_step_total} steps, {n_step_total / duration:0.0f} steps/s")
        y = calculate_smoothed_cumulative_reward(rewards)
        print(f"Smoothed reward for last 100 episodes: {np.mean(y[-100:]):0.1f}")
        i = np.array(y, dtype=np.float64) > 2.5
        if np.sum(i) == 0:
            print(f"Smoothed reward didn't exceed 2.5")
        else:
            i_first = np.where(i)[0][0]
            print(f"Smoothed reward exceed 2.5 at episode {i_first}")
        ys.append(y)

    plt.figure()
    for y in ys:
        plt.plot(y, "b-", linewidth=1)
    y_all = np.vstack(ys).T
    y_avg = np.nanmean(np.array(y_all, dtype=np.float64), axis=1)
    plt.plot(y_avg, "r-", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Smoothed cumulative reward")
    plt.title(f"{name}")
    plt.savefig(f"{name}.jpg", dpi=600)

    return y_avg


class StateHashingXY(StateHashing):
    def hash(self, states):
        agent_xy = states[:, [0, 1]]
        hashes = torch.round(agent_xy).to(torch.int)
        return [(h[0].item(), h[1].item()) for h in hashes]


def main():
    max_reward = 10
    env = GridWorldEnv(size=GRID_SIZE,
                       n_obstacles=len(OBSTACLES),
                       fixed_agent_start_point=(0, 0),
                       fixed_target_point=TARGET,
                       fixed_obstacles=OBSTACLES,
                       target_reward=max_reward,
                       obstacle_reward=-max_reward,
                       default_reward=-0.01)

    n_actions = env.action_space.n
    state, _ = env.reset()
    n_observations = len(state)

    param = Parameters()
    param.obs_dim = n_observations
    param.action_dim = n_actions
    param.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {param.device} as device")

    train_param = get_train_param()
    y1 = run_test(param, train_param, env, "Standard")

    train_param = get_train_param()
    train_param.sampling_strategy = PrioritizedSamplingStrategy(batch_size=BATCH_SIZE)
    train_param.sample_priory_update = PolynomialSamplePriority(10, 1, 0.5)
    y2 = run_test(param, train_param, env, "Priority sampling")

    train_param = get_train_param()
    # bonus_reward_coefficient_scheduler = ConstValueScheduler(0.05)
    bonus_reward_coefficient_scheduler = LinearScheduler(slope=-0.1 / 200, start_value=0.1, min_value=0)
    # hashing = PCAStateHashing(n_components=2, factor=1)
    # counter = ModelHashedStateCounter(hashing)
    counter = SimpleHashedStateCounter(StateHashingXY())
    train_param.count_based_exploration = CountBasedExploration(counter,
                                                                bonus_reward_coefficient_scheduler)
    y3 = run_test(param, train_param, env, "Count based exploration")

    param.n_nets = 7
    train_param = get_train_param()
    train_param.random_action_scheduler = ConstValueScheduler(0.5)
    y4 = run_test(param, train_param, env, "Ensemble")

    plt.figure()
    plt.plot(y1, "-", linewidth=2, label="Standard")
    plt.plot(y2, "-", linewidth=2, label="Prioritized sampling")
    plt.plot(y3, "-", linewidth=2, label="Count based exploration")
    plt.plot(y4, "-", linewidth=2, label="Ensemble")
    plt.xlabel("Episode")
    plt.ylabel("Smoothed cumulative reward")
    plt.title("Method comparison")

    plt.legend()
    plt.savefig("Method comparison.jpg", dpi=600)
    plt.show()


if __name__ == "__main__":
    main()
