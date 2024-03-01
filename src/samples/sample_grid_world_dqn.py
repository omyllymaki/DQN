import logging
import time

import numpy as np
import torch
from matplotlib import pyplot as plt

from src.custom_environments.grid_world_env import GridWorldEnv
from src.dqn.data_buffer import DataBuffer
from src.dqn.dqn_agent import DQNAgent
from src.dqn.parameters import Parameters, TrainParameters
from src.dqn.progress_callback import ProgressCallback, ProgressCallbackVisLatestRewards, ProgressCallbackVisSumReward
from src.dqn.scheduler import ExpDecayScheduler, ConstValueScheduler, LinearScheduler
from src.dqn.utils import running_mean

logging.basicConfig(level=logging.INFO)

GRID_SIZE = 30


class ProgressCallbackGridWorld(ProgressCallback):

    def __init__(self, vis_period=50, n_episodes_to_show=10) -> None:
        super().__init__()
        self.data = []
        self.vis_period = vis_period
        self.n_episodes_to_show = n_episodes_to_show

    def push(self, data: DataBuffer) -> None:
        self.data.append(data.get_all())

    def apply(self) -> None:
        if len(self.data) % self.vis_period != 0:
            return

        plt.subplot(1, 2, 1)
        plt.cla()
        plt.subplot(2, 2, 2)
        plt.cla()
        plt.subplot(2, 2, 4)
        plt.cla()

        last_episodes = self.data[-self.n_episodes_to_show:]
        n_found_target = 0
        n_hit_obstacle = 0
        n_terminated = 0
        for data in last_episodes:

            xs, ys = [], []
            for step_count, item in enumerate(data):
                state = item.state.view(-1).cpu().numpy()
                x = state[0]
                y = state[1]
                xs.append(x)
                ys.append(y)

            last_item = data[-1]
            state = last_item.state.view(-1).cpu().numpy()
            x_target = state[2]
            y_target = state[3]
            objects = state[4:]

            last_reward = last_item.reward.item()
            if last_reward > 0:
                n_found_target += 1
            elif last_reward < -1:
                n_hit_obstacle += 1
            else:
                n_terminated += 1

            plt.subplot(1, 2, 1)
            for i in range(0, len(objects), 2):
                xo = objects[i]
                yo = objects[i + 1]
                plt.plot(xo, yo, "ko")
            plt.plot(xs, ys, "-")
            plt.plot(x_target, y_target, "ro")

            plt.title(f"Found target {n_found_target}, hit obstacle {n_hit_obstacle}, terminated {n_terminated}")

        reward_sums = []
        durations = []
        for data in self.data:
            rewards = [i.reward.item() for i in data]
            reward_sums.append(np.sum(rewards))
            durations.append(len(data))

        plt.xlim(-1, GRID_SIZE + 1)
        plt.ylim(-1, GRID_SIZE + 1)

        plt.subplot(2, 2, 2)
        plt.plot(reward_sums, color="b")
        plt.plot(running_mean(reward_sums, 50), linewidth=2, color="r")
        plt.title("Cumulative reward")
        plt.xlabel("Episode")

        plt.subplot(2, 2, 4)
        plt.plot(durations, color="b")
        plt.plot(running_mean(durations, 50), linewidth=2, color="r")
        plt.title("Duration")
        plt.xlabel("Episode")

        plt.pause(0.1)


def main():
    fixed_obstacles = (
    (11, 24), (13, 2), (7, 11), (4, 8), (24, 5), (19, 20), (5, 15), (13, 24), (8, 17), (28, 7), (0, 22), (4, 1),
    (17, 2), (19, 21), (4, 0))
    env = GridWorldEnv(size=GRID_SIZE,
                       n_obstacles=15,
                       # fixed_agent_start_point=(0, 0),
                       fixed_target_point=(15, 15),
                       fixed_obstacles=fixed_obstacles)

    n_actions = env.action_space.n
    state, _ = env.reset()
    n_observations = len(state)

    param = Parameters()
    param.obs_dim = n_observations
    param.action_dim = n_actions
    param.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {param.device} as device")

    train_param = TrainParameters()
    train_param.discount_scheduler = ConstValueScheduler(0.9)
    train_param.n_episodes = 1000
    train_param.max_steps_per_episode = 200
    train_param.target_network_update_rate = 0.01
    train_param.progress_cb = ProgressCallbackVisLatestRewards(50)
    train_param.progress_cb = ProgressCallbackVisSumReward(50)
    train_param.progress_cb = ProgressCallbackGridWorld(vis_period=10, n_episodes_to_show=10)
    train_param.eps_scheduler = LinearScheduler(slope=-1 / 700, start_value=1.0, min_value=0)

    agent = DQNAgent(param)

    t1 = time.time()
    agent.train(env, train_param)
    t2 = time.time()
    duration = t2 - t1
    n_step_total = agent.steps_done_total
    print(f"Training took {duration} s for {n_step_total} steps, {n_step_total / duration:0.0f} steps/s")

    n_test_runs = 30
    for k in range(n_test_runs):
        results = agent.run(env, 50)

        plt.figure(2)
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
            plt.xlim(-1, GRID_SIZE + 1)
            plt.ylim(-1, GRID_SIZE + 1)
            plt.title(f"Test run {k + 1}/{n_test_runs}")
            plt.pause(0.01)

    plt.show()


if __name__ == "__main__":
    main()
