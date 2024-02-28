import logging
import math
import time
from functools import partial

import gymnasium as gym
import numpy as np
import torch
from matplotlib import pyplot as plt

from src.dqn.dqn_agent import DQNAgent
from src.dqn.parameters import Parameters, TrainParameters
from src.custom_environments.grid_world_env import GridWorldEnv

logging.basicConfig(level=logging.INFO)


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


def plot_search_step_by_step(outcomes):
    if len(outcomes) % 50 != 0:
        return

    xs, ys = [], []
    last_episode = outcomes[-1]
    for step_count, item in enumerate(last_episode):
        observation, reward, terminated, truncated, _ = item

        x = observation[0]
        y = observation[1]
        xs.append(x)
        ys.append(y)
        x_target = observation[2]
        y_target = observation[3]
        objects = observation[4:]

        plt.cla()
        for i in range(0, len(objects), 2):
            xo = objects[i]
            yo = objects[i + 1]
            plt.plot(xo, yo, "ko")

        plt.plot(xs, ys, "b-")
        plt.plot(x, y, "bo")
        plt.plot(x_target, y_target, "ro")
        plt.title(f"Episode {len(outcomes)}, step {step_count}, reward {reward}")
        plt.pause(0.001)
    plt.pause(0.01)


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

    train_param = TrainParameters()
    train_param.eps_update_func = partial(eps_exp_decay_update, decay=1e5)
    train_param.discount_factor_update_func = discount_factor_update
    train_param.episode_results_func = lambda episode_steps: episode_steps[-1][1]  # Reward
    train_param.vis_cb = plot
    train_param.n_episodes = 1000
    train_param.max_steps_per_episode = 200

    # train_param.episode_results_func = lambda episode_steps: episode_steps
    # train_param.vis_cb = plot_search_step_by_step

    xs = np.arange(1, train_param.n_episodes * train_param.max_steps_per_episode)
    eps = [eps_exp_decay_update(x) for x in xs]
    plt.plot(xs, eps)
    plt.ylabel("Epsilon")
    plt.xlabel("Steps")
    plt.show()

    agent = DQNAgent(param)

    t1 = time.time()
    agent.train(env, train_param)
    t2 = time.time()
    duration = t2 - t1
    n_step_total = agent.steps_done
    print(f"Training took {duration} s for {n_step_total} steps, {n_step_total / duration:0.0f} steps/s")

    results = agent.run(env, 200)

    plt.figure()
    xs, ys = [], []
    for step_count, item in enumerate(results):
        observation, reward, terminated, truncated, _ = item

        x = observation[0]
        y = observation[1]
        xs.append(x)
        ys.append(y)
        x_target = observation[2]
        y_target = observation[3]
        objects = observation[4:]

        plt.cla()
        for i in range(0, len(objects), 2):
            xo = objects[i]
            yo = objects[i + 1]
            plt.plot(xo, yo, "ko")

        plt.plot(xs, ys, "b-")
        plt.plot(x, y, "bo")
        plt.plot(x_target, y_target, "ro")
        plt.title(f"Round {step_count}")
        plt.pause(0.1)

    plt.title(f"Agent took {len(results)} steps")
    plt.show()


if __name__ == "__main__":
    main()
