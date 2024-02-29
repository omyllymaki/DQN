import logging

import numpy as np
import torch

from src.custom_environments.grid_world_env import GridWorldEnv
from src.dqn.agent_optimizer import AgentOptimizer
from src.dqn.dqn_agent import DQNAgent
from src.dqn.parameters import Parameters, TrainParameters
from src.dqn.progress_callback import ProgressCallbackVisLatestRewards
from src.dqn.scheduler import ConstValueScheduler, LinearScheduler

logging.basicConfig(level=logging.WARNING)


def score_func(rewards):
    last_episodes = rewards[-50:]
    last_step_rewards = [item[-1] for item in last_episodes]
    return np.mean(last_step_rewards)


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
    base_train_param.discount_scheduler = ConstValueScheduler(0.9)
    base_train_param.n_episodes = 300
    base_train_param.max_steps_per_episode = 200
    base_train_param.eps_scheduler = LinearScheduler(slope=-1 / base_train_param.n_episodes,
                                                     start_value=1.0,
                                                     min_value=0.01)
    base_train_param.progress_cb = ProgressCallbackVisLatestRewards()

    options = {
        "discount_scheduler":
            [
                ConstValueScheduler(0.95),
                ConstValueScheduler(0.9),
                ConstValueScheduler(0.5)
            ],
        "target_network_update_rate": [1e-4, 1e-3, 1e-2],
        "learning_rate": [1e-5, 1e-4, 1e-3],
    }

    agent = DQNAgent(param)
    agent_optimizer = AgentOptimizer(agent)
    results = agent_optimizer.optimize(base_train_param, options, env, 50, score_func)

    scores = [r.score for r in results]
    i_max = np.argmax(scores)
    best_result = results[i_max]

    print(f"Best parameters: {best_result.param}")
    print(f"Score for best parameters: {best_result.score}")


if __name__ == "__main__":
    main()
