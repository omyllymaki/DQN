import logging
import time

import numpy as np
import torch
from matplotlib import pyplot as plt

from src.custom_environments.grid_world_env import GridWorldEnv
from src.dqn.memory import Memory
from src.dqn.dqn_agent import DQNAgent
from src.dqn.parameters import Parameters, TrainParameters
from src.dqn.progress_callback import ProgressCallback
from src.dqn.sampling_strategy import RewardFreqBasedSamplingStrategy
from src.dqn.scheduler import LinearScheduler
from src.dqn.utils import running_mean

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


class ProgressCallbackGridWorld(ProgressCallback):

    def __init__(self, vis_period=50, n_episodes_to_show=10) -> None:
        super().__init__()
        self.data = []
        self.vis_period = vis_period
        self.n_episodes_to_show = n_episodes_to_show
        self.heatmap = np.zeros((GRID_SIZE, GRID_SIZE))

    def push(self, data: Memory) -> None:
        self.data.append(data.get_all())
        states = [item.state for item in data]
        for state in states:
            x = int(state[0][0].item())
            y = int(state[0][1].item())
            self.heatmap[x, y] += 1

    def apply(self) -> None:
        if len(self.data) % self.vis_period != 0:
            return

        plt.figure(1)
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

        plt.figure(2)
        plt.cla()
        ub = np.percentile(self.heatmap, 95)
        lb = np.percentile(self.heatmap, 5)
        plt.imshow(self.heatmap.T, vmin=lb, vmax=ub)
        plt.gca().invert_yaxis()
        plt.plot(TARGET[0], TARGET[1], "ro")

        for obs in OBSTACLES:
            plt.plot(obs[0], obs[1], "bo")

        plt.pause(0.1)

        # self.heatmap = np.zeros((GRID_SIZE, GRID_SIZE))


def state_hash_func(state):
    x = state[0][0].item()
    y = state[0][1].item()
    return int(x), int(y)


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
    train_param.progress_cb = ProgressCallbackGridWorld(vis_period=10, n_episodes_to_show=10)
    train_param.eps_scheduler = LinearScheduler(slope=-1 / 700, start_value=1.0, min_value=0)
    train_param.exploration_bonus_reward_coeff_scheduler = LinearScheduler(slope=-1 / 700, start_value=1.0, min_value=0)
    train_param.state_hash_func = state_hash_func

    agent = DQNAgent(param)

    t1 = time.time()
    agent.train(env, train_param)
    t2 = time.time()
    duration = t2 - t1
    n_step_total = agent.steps_done_total
    print(f"Training took {duration} s for {n_step_total} steps, {n_step_total / duration:0.0f} steps/s")

    plt.show()


if __name__ == "__main__":
    main()
