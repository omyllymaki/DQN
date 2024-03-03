from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

from src.dqn.model import FNN
from src.dqn.progress_callback import ProgressCallbackSimple
from src.dqn.sampling_strategy import SamplingStrategy, RandomSamplingStrategy
from src.dqn.scheduler import ExpDecayScheduler, ConstValueScheduler


@dataclass
class Parameters:
    """
    Collection of parameters for the agent.

    Attributes:
    obs_dim: Observation vector size.
    action_dim: Action vector size.
    n_nets: Number of models.
    net: Model to learn Q function.
    device: Device used.
    """
    obs_dim = None
    action_dim = None
    net = FNN
    n_nets = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainParameters:
    """
    Collection of parameters for training the agent.

    Attributes:
    optimizer: Any pytorch optimizer.
    loss: Any pytorch loss.
    n_episodes: Number od episodes to run.
    max_steps_per_episode: Maximum number of steps the agent can take during one episode.
    buffer_size: Size of the memory used to sample data for updates.
    learning_rate: Optimizer learning rate.
    sampling_strategy: Strategy to draw samples from replay memory.
    discount_scheduler: Scheduler to update discount factors during training. Discount factor determines the importance of future rewards relative to immediate rewards.
    eps_scheduler: Scheduler to update eps values during training. eps is probability to select "exploration" instead of best action.
    random_action_scheduler: Scheduler to update random action probablity. This determines choice between random action and getting most uncertain action in "exploration".
    state_hash_func: function to map state to hash. Hash is used to count for every state hash, and this is used to calculate exploration bonus reward.
    exploration_bonus_reward_coeff_scheduler: Coefficient used to calculate bonus reward based on hashed state counts.
    target_network_update_rate: Coefficient to update target network using exponential moving average.
    gradient_clipping: Gradient clipping applied in policy net updates.
    progress_cb: progress_cb is called after every episode and does defined steps to the data. See ProgressCallback for more information.

    """
    optimizer = optim.AdamW
    loss = nn.SmoothL1Loss
    max_steps_per_episode = 1000
    n_episodes = 500
    buffer_size = 10000
    learning_rate = 1e-4
    sampling_strategy = RandomSamplingStrategy(batch_size=128)
    discount_scheduler = ConstValueScheduler(0.9)
    eps_scheduler = ExpDecayScheduler(start=0.9, end=0.05, decay=10000)
    random_action_scheduler = ConstValueScheduler(1.0)
    state_hash_func = None
    exploration_bonus_reward_coeff_scheduler = ConstValueScheduler(0)
    target_network_update_rate = 0.005
    gradient_clipping = 100
    progress_cb = ProgressCallbackSimple()
