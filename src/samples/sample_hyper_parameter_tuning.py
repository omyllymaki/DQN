import logging
import math
from functools import partial

import numpy as np
import torch
from matplotlib import pyplot as plt

from src.dqn.agent_optimizer import AgentOptimizer
from src.dqn.dqn_agent import DQNAgent
from src.dqn.parameters import Parameters, TrainParameters
from src.custom_environments.grid_world_env import GridWorldEnv

logging.basicConfig(level=logging.WARNING)


def running_mean(x, windwow_size):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    output = (cumsum[windwow_size:] - cumsum[:-windwow_size]) / float(windwow_size)
    n_padding = len(x) - len(output)
    return n_padding * [None] + output.tolist()


def plot(outcomes):
    plt.figure(1)
    plt.cla()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(outcomes, "b-")
    plt.plot(running_mean(outcomes, 50), "r-", linewidth=2)
    plt.plot()
    plt.pause(0.001)


def eps_exp_decay_update(steps_done, start=0.9, end=0.05, decay=10000):
    return end + (start - end) * math.exp(-1. * steps_done / decay)


def discount_factor_update(steps_done):
    return 0.9


def main():
    env = GridWorldEnv(size=20, n_obstacles=0)

    n_actions = env.action_space.n
    state, _ = env.reset()
    n_observations = len(state)

    param = Parameters()
    param.obs_dim = n_observations
    param.action_dim = n_actions
    param.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {param.device} as device")

    base_train_param = TrainParameters()
    base_train_param.eps_update_func = partial(eps_exp_decay_update, decay=1e5)
    base_train_param.discount_factor_update_func = discount_factor_update
    base_train_param.episode_results_func = lambda episode_steps: episode_steps[-1][1]  # Reward
    base_train_param.vis_cb = plot
    base_train_param.n_episodes = 300
    base_train_param.max_steps_per_episode = 200

    options = {
        "eps_update_func":
            [
                partial(eps_exp_decay_update, decay=100),
                partial(eps_exp_decay_update, decay=1000),
                partial(eps_exp_decay_update, decay=10000)
            ],
        "target_network_update_rate": [1e-4, 1e-3, 1e-2],
        "learning_rate": [1e-5, 1e-4, 1e-3],
    }

    agent = DQNAgent(param)
    agent_optimizer = AgentOptimizer(agent)
    results = agent_optimizer.optimize(base_train_param, options, env, 10)

    scores = [r.score for r in results]
    i_max = np.argmax(scores)
    best_result = results[i_max]

    print(f"Best parameters: {best_result.param}")
    print(f"Score for best parameters: {best_result.score}")


if __name__ == "__main__":
    main()
