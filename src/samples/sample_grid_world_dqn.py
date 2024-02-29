import logging
import time

import numpy as np
import torch
from matplotlib import pyplot as plt

from src.custom_environments.grid_world_env import GridWorldEnv
from src.dqn.data_buffer import DataBuffer
from src.dqn.dqn_agent import DQNAgent
from src.dqn.parameters import Parameters, TrainParameters
from src.dqn.progress_callback import ProgressCallback, ProgressCallbackVisLatestRewards
from src.dqn.scheduler import ExpDecayScheduler, ConstValueScheduler

logging.basicConfig(level=logging.INFO)


class ProgressCallbackVisualizeSearch(ProgressCallback):

    def __init__(self, vis_period=10) -> None:
        super().__init__()
        self.data = None
        self.vis_period = vis_period
        self.push_counter = 0

    def push(self, data: DataBuffer) -> None:
        self.data = data.get_all()
        self.push_counter += 1

    def apply(self) -> None:
        if self.push_counter % self.vis_period != 0:
            return

        xs, ys = [], []
        for step_count, item in enumerate(self.data):

            state = item.state.view(-1).cpu().numpy()
            x = state[0]
            y = state[1]
            x_target = state[2]
            y_target = state[3]
            objects = state[4:]
            reward = item.reward.item()

            xs.append(x)
            ys.append(y)

            plt.cla()
            for i in range(0, len(objects), 2):
                xo = objects[i]
                yo = objects[i + 1]
                plt.plot(xo, yo, "ko")

            plt.plot(xs, ys, "b-")
            plt.plot(x, y, "bo")
            plt.plot(x_target, y_target, "ro")
            plt.title(f"Episode {self.push_counter}, step {step_count}, reward {reward}")
            plt.pause(0.001)
        plt.pause(0.5)


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
    train_param.eps_scheduler = ExpDecayScheduler(start=0.9, end=0.05, decay=1e4)
    train_param.discount_scheduler = ConstValueScheduler(0.9)
    train_param.n_episodes = 1000
    train_param.max_steps_per_episode = 200
    train_param.progress_cb = ProgressCallbackVisLatestRewards()
    # train_param.progress_cb = ProgressCallbackVisualizeSearch()

    xs = np.arange(1, train_param.n_episodes * train_param.max_steps_per_episode)
    eps = [train_param.eps_scheduler.apply(0, 0, x) for x in xs]
    plt.plot(xs, eps)
    plt.ylabel("Epsilon")
    plt.xlabel("Steps")
    plt.show()

    agent = DQNAgent(param)

    t1 = time.time()
    agent.train(env, train_param)
    t2 = time.time()
    duration = t2 - t1
    n_step_total = agent.steps_done_total
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
