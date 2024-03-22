import argparse
import json
import logging
import os
import time
from copy import deepcopy
from pprint import pprint

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

METHOD_CHOICES = ["standard", "priority_sampling", "count_based_exploration", "ensemble"]
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


def get_param(args, n_actions, n_observations):
    param = Parameters()
    param.obs_dim = n_observations
    param.action_dim = n_actions
    param.device = args.device
    return param


def get_train_param(args):
    train_param = TrainParameters()
    train_param.n_episodes = args.n_episodes
    train_param.max_steps_per_episode = args.max_steps_per_episode
    train_param.target_network_update_rate = args.target_network_update_rate

    episodes_to_zero_value = int(args.exploration_duration * args.n_episodes)
    slope = -args.eps_scheduler_start_value / episodes_to_zero_value
    train_param.eps_scheduler = LinearScheduler(slope=slope,
                                                start_value=args.eps_scheduler_start_value,
                                                min_value=0)
    train_param.learning_rate_scheduler = ConstValueScheduler(args.learning_rate)
    train_param.progress_cb = None
    train_param.sampling_strategy = RandomSamplingStrategy(batch_size=args.batch_size)
    train_param.buffer_size = args.buffer_size
    return train_param


def calculate_smoothed_cumulative_reward(rewards, window_size=50):
    return running_mean([np.sum(r) for r in rewards], window_size)


def run_test(param, train_param, env, name, args):
    print(f"Start training: {name}")
    ys, all_rewards = [], []
    fig, ax = plt.subplots()
    for k in range(args.n_runs):
        print(f"Run number {k + 1}/{args.n_runs}")
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
        all_rewards.append(rewards)

        y_all = np.vstack(ys).T
        y_avg = np.nanmean(np.array(y_all, dtype=np.float64), axis=1)
        y_avg = np.where(np.isnan(y_avg), None, y_avg).tolist()

        ax.cla()
        for y in ys:
            ax.plot(y, "b-", linewidth=1)
        ax.plot(y_avg, "r-", linewidth=2)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Smoothed cumulative reward")
        ax.set_title(f"{name}")
        plt.pause(0.5)

        fig.savefig(f"{name}.jpg", dpi=600)
        save_results(args, y_avg, ys, rewards, name)


class StateHashingXY(StateHashing):
    def hash(self, states):
        agent_xy = states[:, [0, 1]]
        hashes = torch.round(agent_xy).to(torch.int)
        return [(h[0].item(), h[1].item()) for h in hashes]


def create_parser():
    # @formatter:off
    parser = argparse.ArgumentParser(description="Arguments for training parameters")
    parser.add_argument("--device", type=str, default="cpu", help="Device used for model")
    parser.add_argument("--method", type=str, default="standard", choices=METHOD_CHOICES, help="Method used")
    parser.add_argument("--n_episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--max_steps_per_episode", type=int, default=200, help="Maximum steps per episode")
    parser.add_argument("--target_network_update_rate", type=float, default=0.1, help="Target network update rate")
    parser.add_argument("--eps_scheduler_start_value", type=float, default=0.7, help="Epsilon scheduler start value")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for policy net updates")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--buffer_size", type=int, default=15000, help="Memory buffer size")
    parser.add_argument("--n_runs", type=int, default=10, help="Number of runs per method")
    parser.add_argument("--exploration_duration", type=float, default=0.7, help="Relative duration of exploration phase")

    parser.add_argument("--count_based_exploration-bonus_reward_start_value", type=float, default=0.1, help="Bonus reward scheduler start value used in count based exploration")

    parser.add_argument("--sample_priority-max_tde_error", type=float, default=10, help="Sample priority calculation max TDE error")
    parser.add_argument("--sample_priority-beta", type=float, default=1, help="Sample priority calculation power coefficient")
    parser.add_argument("--sample_priority-alpha", type=float, default=0.5, help="Sample priority calculation EMA coefficient")

    parser.add_argument("--ensemble-n_models", type=int, default=7, help="Number of models in ensemble learning")
    parser.add_argument("--ensemble-random_action_prob", type=float, default=0.5, help="Probability for random action in ensemble learning")
    # @formatter:on
    return parser


def save_json(path, data):
    with open(path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def save_results(args, smoothed_rewards_avg, smoothed_rewards, rewards, name):
    output = {
        "name": name,
        "parameters": vars(args),
        "smoothed_rewards_avg": smoothed_rewards_avg,
        "smoothed_rewards": smoothed_rewards,
        "rewards": rewards
    }
    save_json(name + ".json", output)


def main():
    parser = create_parser()
    args = parser.parse_args()
    pid = os.getpid()
    print(f"PID: {pid}")
    print(f"Arguments: {args}")

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

    if args.method == "standard":
        param = get_param(args, n_actions, n_observations)
        train_param = get_train_param(args)
    elif args.method == "priority_sampling":
        param = get_param(args, n_actions, n_observations)
        train_param = get_train_param(args)
        train_param.sampling_strategy = PrioritizedSamplingStrategy(batch_size=args.batch_size)
        train_param.sample_priory_update = PolynomialSamplePriority(args.sample_priority_max_tde_error,
                                                                    args.sample_priority_beta,
                                                                    args.sample_priority_alpha)
    elif args.method == "count_based_exploration":
        param = get_param(args, n_actions, n_observations)
        train_param = get_train_param(args)
        episodes_to_zero_value = int(args.exploration_duration * args.n_episodes)
        start_value = args.count_based_exploration_bonus_reward_start_value
        slope = -start_value / episodes_to_zero_value
        bonus_reward_coefficient_scheduler = LinearScheduler(slope=-slope,
                                                             start_value=start_value,
                                                             min_value=0)
        counter = SimpleHashedStateCounter(StateHashingXY())
        train_param.count_based_exploration = CountBasedExploration(counter,
                                                                    bonus_reward_coefficient_scheduler)
    elif args.method == "ensemble":
        param = get_param(args, n_actions, n_observations)
        train_param = get_train_param(args)
        param.n_nets = args.ensemble_n_models
        train_param.random_action_scheduler = ConstValueScheduler(args.ensemble_random_action_prob)
    else:
        raise ValueError(f"Invalid argument: '{args.method}'. Allowed choices are {METHOD_CHOICES}")

    name = f"{args.method} {pid}"
    run_test(param, train_param, env, name, args)
    plt.show()


if __name__ == "__main__":
    main()
