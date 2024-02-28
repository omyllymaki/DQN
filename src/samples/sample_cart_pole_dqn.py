import logging
import math
import time

import gymnasium as gym
import numpy as np
import torch
from matplotlib import pyplot as plt

from src.dqn.dqn_agent import DQNAgent
from src.dqn.parameters import Parameters, TrainParameters

logging.basicConfig(level=logging.INFO)


def plot(outcomes):
    plt.figure(1)
    plt.cla()
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(outcomes)
    plt.pause(0.001)


def eps_exp_decay_update(steps_done, start=0.9, end=0.05, decay=1000):
    return end + (start - end) * math.exp(-1. * steps_done / decay)


def discount_factor_update(steps_done):
    return 0.9


def main():
    env = gym.make("CartPole-v1")

    n_actions = env.action_space.n
    state, _ = env.reset()
    n_observations = len(state)

    param = Parameters()
    param.obs_dim = n_observations
    param.action_dim = n_actions
    param.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {param.device} as device")

    train_param = TrainParameters()
    train_param.eps_update_func = eps_exp_decay_update
    train_param.discount_factor_update_func = discount_factor_update
    train_param.episode_results_func = lambda episode_steps: len(episode_steps)
    train_param.vis_cb = plot
    train_param.n_episodes = 300
    train_param.max_steps_per_episode = 500

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

    results = agent.run(env, 500)

    plt.figure()
    for i, item in enumerate(results):
        observation, reward, terminated, truncated, _ = item

        x = observation[0]
        angle = observation[2] + np.pi / 2
        y = 0
        xd = x + np.cos(angle)
        yd = y + np.sin(angle)

        plt.cla()
        plt.plot(x, y, "ro")
        plt.plot([x, xd], [y, yd], "b-")
        plt.xlim(-3, 3)
        plt.ylim(-0.5, 1.5)
        plt.title(f"Round {i}")
        plt.pause(0.1)

    plt.title(f"Agent took {len(results)} steps")
    plt.show()


if __name__ == "__main__":
    main()
