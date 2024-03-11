import logging
import random
import time
from statistics import mode
from typing import Optional, List, Tuple

import gym
import numpy as np
import torch
from matplotlib import pyplot as plt

from src.dqn.memory import Memory
from src.dqn.parameters import Parameters, TrainParameters
from src.dqn.progress_data import ProgressData
from src.dqn.scheduler import Stage

logger = logging.getLogger(__name__)


class DQNAgent:
    """
    The implementation is quite standard DQN, including
        - Simple feedforward network to approximate state-value function
        - Replay memory
        - Separate target network that is "smoothed" version of policy net

    It also contains some experimental features:
        - Option for ensemble learning
          - Voting best action based on multiple models
          - Finding most uncertain actions in exploration phase using multiple models
        - Option for adding reward bonus based on (hashed) state count as exploration strategy
        - Option to use priority based sampling from replay memory; priority defined by temporal difference error
    """

    def __init__(self, parameters: Parameters) -> None:
        self.param = parameters
        self.policy_nets = []
        self.target_nets = []
        self.train_param = None
        self.optimizers = []
        self.memory = None
        self.eps = None
        self.p_random_action = None
        self.discount_factor = None
        self.dropout = None
        self.bonus_reward_coeff = None
        self.stage = None
        self.reset()

    def reset(self) -> None:
        for _ in range(self.param.n_nets):
            policy_net = self.param.net(self.param.obs_dim, self.param.action_dim)
            target_net = self.param.net(self.param.obs_dim, self.param.action_dim)
            target_net.load_state_dict(policy_net.state_dict())  # Set same init weights as in policy_net
            policy_net.to(self.param.device)
            target_net.to(self.param.device)
            self.policy_nets.append(policy_net)
            self.target_nets.append(target_net)

        self.train_param = None
        self.optimizers = None
        self.memory = None
        self.eps = None
        self.discount_factor = None
        self.bonus_reward_coeff = None
        self.stage = None

    def train(self, env: gym.Env, train_param: TrainParameters) -> List[List[float]]:
        """
        Train agent using environment env and training paramaters train_param.

        Training process:

        init replay memory
        for every episode:
            reset env
            get init state
            for as long as terminated:
                scheduled parameter updates
                select action based on learned policies or randomly, based on eps value
                perform action and receive feedback from env
                push data to replay memory
                update policy model
                 - draw sample from replay memory based on selected strategy
                 - add bonus reward based on state counts (optional)
                 - calculate expected state action values using target model
                 - update sample priorities in the replay memory based on temporal difference errors (optional)
                 - minimize temporal difference errors by optimizing policy net weights
                update target net based on policy net weights

        Args:
            env (gym.Env): environment which interacts with the agent.
            train_param (TrainParameters): training parameters.

        Returns:
            rewards: all the rewards collected during training.
        """
        self.train_param = train_param
        self.optimizers = []
        for policy_net in self.policy_nets:
            optimizer = train_param.optimizer(policy_net.parameters(),
                                              lr=self.train_param.learning_rate,
                                              amsgrad=True)
            self.optimizers.append(optimizer)
        self.memory = Memory(train_param.buffer_size)
        self.stage = Stage()

        rewards = []
        for i_episode in range(train_param.n_episodes):
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.param.device).unsqueeze(0)
            episode_data = ProgressData()
            episode_rewards = []
            for step_counter in range(self.train_param.max_steps_per_episode):
                self.stage.i_episode = i_episode
                self.stage.n_steps_episode = step_counter
                self.stage.n_steps_total += 1
                self._scheduled_updates()

                action = self._select_action(state, env)
                step_result = env.step(action.item())
                logger.debug(f"Episode {i_episode}, step count {step_counter}, {step_result}")
                observation, reward, terminated, truncated, _ = step_result
                episode_rewards.append(reward)
                reward = torch.tensor([reward], device=self.param.device)
                next_state = torch.tensor(observation, dtype=torch.float32, device=self.param.device).unsqueeze(0)
                episode_data.push_transition(state.cpu(),
                                             action.cpu(),
                                             next_state.cpu(),
                                             reward.cpu())

                done = terminated or truncated
                if terminated:
                    next_state = None

                self.memory.push(state, action, next_state, reward)
                if self.train_param.count_based_exploration is not None:
                    self.train_param.count_based_exploration.push(state)
                state = next_state

                loss_list, temporal_diff_errors_list = [], []
                if len(self.memory) >= self.train_param.sampling_strategy.batch_size:
                    for policy_net, target_net, optimizer in zip(self.policy_nets, self.target_nets, self.optimizers):
                        loss, temporal_diff_errors = self._update_policy_model(policy_net, target_net, optimizer)
                        loss_list.append(loss.item())
                        temporal_diff_errors_list.append(temporal_diff_errors.cpu())
                        logger.debug(f"Episode {i_episode}, step count {step_counter}, loss {loss}")
                        self._update_target_net(policy_net, target_net)
                episode_data.push_losses(loss_list)
                episode_data.push_temporal_difference_errors(temporal_diff_errors_list)

                if done:
                    if terminated:
                        logger.info(f"The episode {i_episode} was terminated after {step_counter} steps")
                    if truncated:
                        logger.info(f"The episode {i_episode} was truncated after {step_counter} steps")
                    break

                if step_counter == self.train_param.max_steps_per_episode - 1:
                    logger.info(
                        f"The episode {i_episode} was ended because max number of steps {step_counter + 1} was reached")

            rewards.append(episode_rewards)
            if self.train_param.progress_cb is not None:
                self.train_param.progress_cb.push(episode_data)
                self.train_param.progress_cb.apply()

        return rewards

    def run(self, env: gym.Env, max_steps: int) -> list:
        """
        run agent in env. Agent chooces the actions based on observed states and learned policy.

        Args:
            env (gym.Env): enviroment which interacts with the agent.
            max_steps (int): Maximum number of steps allowed.

        Returns:
            list: feedback from env, every item containing (observation, reward, terminated, truncated, info)
        """
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=self.param.device).unsqueeze(0)
        results = []
        for step_counter in range(max_steps):
            action = self._get_best_action(state)
            step_result = env.step(action.item())
            results.append(step_result)
            observation, reward, terminated, truncated, _ = step_result

            done = terminated or truncated
            if done:
                if terminated:
                    logger.info(f"The run was terminated after {step_counter} steps")
                if truncated:
                    logger.info(f"The run was truncated after {step_counter} steps")
                break

            next_state = torch.tensor(observation, dtype=torch.float32, device=self.param.device).unsqueeze(0)
            state = next_state

            if step_counter == self.train_param.max_steps_per_episode - 1:
                logger.info(
                    f"The run was ended because max number of steps {step_counter + 1} was reached")

        return results

    def _scheduled_updates(self):
        self.eps = self.train_param.eps_scheduler.apply(self.stage)
        self.discount_factor = self.train_param.discount_scheduler.apply(self.stage)
        self.p_random_action = self.train_param.random_action_scheduler.apply(self.stage)

    def _get_best_action(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            best_actions = []
            for policy_net in self.policy_nets:
                q_values = policy_net.forward(state)
                best_action = torch.argmax(q_values)
                best_actions.append(best_action)
            best_action_voted = mode(best_actions)
            output = best_action_voted.view(1, 1)
            return output

    def _get_most_uncertain_action(self, state):
        with torch.no_grad():
            q_values_list = []
            for policy_net in self.policy_nets:
                q_values = policy_net.forward(state)
                q_values_list.append(q_values)
            concatenated_q_values = torch.cat(q_values_list, dim=0)
            std = concatenated_q_values.std(dim=0)
            output = torch.argmax(std)  # Most uncertain action is the one which has the most deviation among the models
            return output.view(1, 1)

    def _select_action(self, state: torch.Tensor, env: gym.Env) -> torch.Tensor:
        # TODO: refactor this. Maybe a separate ActionSelectionStrategy class?
        sample = random.random()
        if sample > self.eps:
            return self._get_best_action(state)
        else:
            p = random.random()
            if p > self.p_random_action:
                return self._get_most_uncertain_action(state)
            else:
                return torch.tensor([[env.action_space.sample()]], device=self.param.device, dtype=torch.long)

    def _update_policy_model(self, policy_net, target_net, optimizer) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, sample_indices = self.train_param.sampling_strategy.apply(self.memory)

        # Compute a mask for not-non values (none means terminated episode)
        is_not_none = torch.tensor([i is not None for i in batch.next_state], device=self.param.device,
                                   dtype=torch.bool)

        # Prepare tensors for the update
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        if self.train_param.count_based_exploration is not None:
            bonus_reward = self.train_param.count_based_exploration.get_bonus_rewards(state_batch, self.stage)
            reward_batch += bonus_reward.to(self.param.device)

        # Compute Q values for the states.
        # Then we select the columns based on actions taken.
        q_values = policy_net(state_batch)  # "Q table", [batch_size, n_actions] tensor
        current_state_action_values = q_values.gather(1, action_batch)  # Q values for every action taken

        # Expected values of actions are computed based on the smoothed target_net by selecting their best Q values for the next states
        # This adds stability, compared to calculating these with policy_net
        # We need to use is_not_none mask here in order to handle none values of the next states (final states)
        action_values_for_next_states = torch.zeros(self.train_param.sampling_strategy.batch_size,
                                                    device=self.param.device)
        with torch.no_grad():
            action_values_for_next_states[is_not_none] = target_net(non_final_next_states).max(1).values
        expected_state_action_values = (action_values_for_next_states * self.discount_factor) + reward_batch
        temporal_diff_errors = current_state_action_values - expected_state_action_values.unsqueeze(1)

        # Update sample priorities in the memory
        # This relevant only is priority based sampling strategy is used
        if self.train_param.sample_priory_update is not None:
            for index, tde in zip(sample_indices, temporal_diff_errors):
                current_priority = self.memory.memory[index].priority
                self.memory.memory[index].priority = self.train_param.sample_priory_update.apply(current_priority,
                                                                                                 tde.item())

        # Minimize temporal difference error by updating weights of the policy_net
        criterion = self.train_param.loss()
        loss = criterion(current_state_action_values, expected_state_action_values.unsqueeze(1))

        # Just regular weight update using gradient descent but with gradient clipping
        optimizer.zero_grad()
        loss.backward()
        if self.train_param.gradient_clipping is not None:
            torch.nn.utils.clip_grad_value_(policy_net.parameters(),
                                            self.train_param.gradient_clipping)  # In-place gradient clipping
        optimizer.step()

        if self.stage.n_steps_total % 500 == 0:
            tde_min = temporal_diff_errors.min().item()
            tde_max = temporal_diff_errors.max().item()

            logger.info(f"Fit Information:\n"
                        f"Stage {self.stage}\n"
                        f"Temporal diff errors: min {tde_min}, max {tde_max}\n"
                        f"Loss: {loss.item()}")

        return loss, temporal_diff_errors

    def _update_target_net(self, policy_net, target_net):
        # Target net weights are updated using exponential moving average
        # New values come from policy net
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        ur = self.train_param.target_network_update_rate
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * ur + target_net_state_dict[key] * (1 - ur)
        target_net.load_state_dict(target_net_state_dict)
