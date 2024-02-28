from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim

from src.dqn.model import FNN


@dataclass
class Parameters:
    """
    Collection of parameters for the agent.

    Attributes:
    obs_dim: Observation vector size.
    action_dim: Action vector size.
    net: Model to learn Q function.
    device: Device used.
    """
    obs_dim = None
    action_dim = None
    net = FNN
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
    batch_size: Batch size for updates.
    discount_factor_update_func: Q learning discount factor update function. Discount factor determines the importance of future rewards relative to immediate rewards.
    eps_update_func: Function to update eps. eps is probability to select random action and drives exploration in the system.
    target_network_update_rate: Coefficient to update target network using exponential moving average.
    gradient_clipping: Gradient clipping applied in policy net updates.
    episode_results_func: Function to collect output for episode data. Input data to this function is list of env.step() outputs from the episode.
    vis_cb: Callback to perform visualization during training. Input to this function is list of outcomes calculated by episode_results_func. vis_cb is called when episode is temrinated.
    """
    optimizer = optim.AdamW
    loss = nn.SmoothL1Loss
    max_steps_per_episode = 1000
    n_episodes = 500
    buffer_size = 10000
    learning_rate = 1e-4
    batch_size = 128
    discount_factor_update_func = lambda self, steps_taken: 0.9
    eps_update_func = lambda self, steps_taken: 0.999 ** steps_taken
    target_network_update_rate = 0.005
    gradient_clipping = 100
    episode_results_func = lambda self, episode_steps: episode_steps
    vis_cb = None
