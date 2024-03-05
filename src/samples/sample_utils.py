import numpy as np
from matplotlib import pyplot as plt

from src.dqn.progress_callback import ProgressCallback
from src.dqn.utils import running_mean


class ProgressCallbackGridWorld(ProgressCallback):

    def __init__(self, grid_size, target, obstacles, vis_period=50, n_episodes_to_show=10) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.target = target
        self.obstacles = obstacles
        self.data = []
        self.vis_period = vis_period
        self.n_episodes_to_show = n_episodes_to_show
        self.heatmap = np.zeros((self.grid_size, self.grid_size))

    def push(self, data) -> None:
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

        plt.xlim(-1, self.grid_size + 1)
        plt.ylim(-1, self.grid_size + 1)

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
        plt.plot(self.target[0], self.target[1], "ro")

        for obs in self.obstacles:
            plt.plot(obs[0], obs[1], "bo")

        plt.pause(0.1)
