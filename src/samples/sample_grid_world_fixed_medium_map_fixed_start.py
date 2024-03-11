import logging
import time

import torch
from matplotlib import pyplot as plt

from src.custom_environments.grid_world.grid_world_env import GridWorldEnv
from src.dqn.count_based_exploration import CountBasedExploration
from src.dqn.counter import SimpleHashedStateCounter, ModelHashedStateCounter
from src.dqn.dqn_agent import DQNAgent
from src.dqn.pca import PCA
from src.dqn.state_hashing import RandomProjectionStateHashing, PCAStateHashing
from src.dqn.parameters import Parameters, TrainParameters
from src.dqn.sampling_strategy import RandomSamplingStrategy, PrioritizedSamplingStrategy
from src.dqn.scheduler import ConstValueScheduler, LinearScheduler
from src.samples.sample_utils import ProgressCallbackGridWorld

logging.basicConfig(level=logging.INFO)

GRID_SIZE = 50
TARGET = (25, 25)

OBSTACLES = (
    (19, 12), (8, 30), (45, 39), (13, 20), (7, 29), (43, 3), (33, 22), (28, 6), (11, 9), (26, 23), (3, 13), (40, 35),
    (5, 28), (12, 9), (30, 33), (34, 38), (37, 20), (23, 16), (33, 26), (31, 9))


class StateHashing:
    def apply(self, state):
        x = state[0][0].item()
        y = state[0][1].item()
        return int(x), int(y)


def main():
    env = GridWorldEnv(size=GRID_SIZE,
                       n_obstacles=len(OBSTACLES),
                       fixed_agent_start_point=(0, 0),
                       fixed_target_point=TARGET,
                       fixed_obstacles=OBSTACLES,
                       target_reward=50,
                       obstacle_reward=-50)

    n_actions = env.action_space.n
    state, _ = env.reset()
    n_observations = len(state)

    param = Parameters()
    param.n_nets = 1
    param.obs_dim = n_observations
    param.action_dim = n_actions
    param.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {param.device} as device")

    train_param = TrainParameters()
    train_param.sampling_strategy = RandomSamplingStrategy(batch_size=128)
    train_param.discount_scheduler = ConstValueScheduler(0.9)
    train_param.n_episodes = 5000
    train_param.max_steps_per_episode = 300
    train_param.target_network_update_rate = 0.005
    train_param.progress_cb = ProgressCallbackGridWorld(grid_size=GRID_SIZE,
                                                        target=TARGET,
                                                        obstacles=OBSTACLES,
                                                        vis_period=10,
                                                        n_episodes_to_show=10)
    train_param.state_hashing = StateHashing()

    train_param.sampling_strategy = RandomSamplingStrategy(batch_size=128)
    train_param.eps_scheduler = LinearScheduler(slope=-0.8 / 3000, start_value=0.8, min_value=0.05)
    bonus_reward_coefficient_scheduler = LinearScheduler(slope=-1.0 / 1500, start_value=1.0, min_value=0.05)
    hashing = PCAStateHashing(n_components=2, factor=1)
    counter = ModelHashedStateCounter(hashing)
    train_param.count_based_exploration = CountBasedExploration(counter,
                                                                bonus_reward_coefficient_scheduler)

    # train_param.count_based_exploration = None

    train_param.sampling_strategy = PrioritizedSamplingStrategy(64)
    train_param.target_network_update_rate = 0.05
    train_param.buffer_size = 50000

    agent = DQNAgent(param)

    t1 = time.time()
    agent.train(env, train_param)
    t2 = time.time()
    duration = t2 - t1
    n_step_total = agent.stage.n_steps_total
    print(f"Training took {duration} s for {n_step_total} steps, {n_step_total / duration:0.0f} steps/s")

    plt.figure()
    n_test_runs = 30
    for k in range(n_test_runs):
        results = agent.run(env, 50)

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
