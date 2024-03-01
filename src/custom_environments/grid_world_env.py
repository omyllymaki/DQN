from typing import Optional, Tuple

import gym
import numpy as np
from gym import spaces


class GridWorldEnv(gym.Env):
    # TODO: add randomness to actions

    def __init__(self,
                 size=10,
                 n_obstacles=3,
                 fixed_agent_start_point: Optional[Tuple[int, int]] = None,
                 fixed_target_point: Optional[Tuple[int, int]] = None,
                 fixed_obstacles: Optional[tuple] = None,
                 target_reward=10, obstacle_reward=-10, default_reward=-0.05,
                 seed=42):
        self.size = size
        self.n_obstacles = n_obstacles
        self.fixed_agent_start_point = fixed_agent_start_point
        self.fixed_target_point = fixed_target_point
        self.fixed_obstacles = fixed_obstacles
        self.target_reward = target_reward
        self.obstacle_reward = obstacle_reward
        self.default_reward = default_reward
        self.seed = seed

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )
        for i in range(n_obstacles):
            self.observation_space[f"obstacle{i}"]: spaces.Box(0, size - 1, shape=(2,), dtype=int)

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        self._get_init_agent_location()
        self._get_world()

    def reset(self, seed=None, options=None):

        # We need the following line to seed self.np_random
        if self.seed is not None:
            super().reset(seed=self.seed)
        else:
            super().reset(seed=seed)

        self._get_init_agent_location()
        self._get_world()

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]

        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        terminated = False
        reward = self.default_reward
        if np.array_equal(self._agent_location, self._target_location):  # Agent reached the target
            terminated = True
            reward = self.target_reward
        for obstacle_location in self._obstacle_locations:
            if np.array_equal(self._agent_location, obstacle_location):  # Agent hits to obstacle
                terminated = True
                reward = self.obstacle_reward

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def _random_loc(self):
        return np.random.randint(0, self.size, size=2)

    def _get_init_agent_location(self):
        if self.fixed_agent_start_point is not None:
            self._agent_location = np.array(self.fixed_agent_start_point)
        else:
            self._agent_location = self._random_loc()

    def _get_world(self):
        if self.fixed_target_point:
            self._target_location = np.array(self.fixed_target_point)
        else:
            while True:
                self._target_location = self._random_loc()
                if np.array_equal(self._target_location, self._agent_location):
                    continue
                break
        if self.fixed_obstacles:
            self._obstacle_locations = [np.array(o) for o in self.fixed_obstacles]
        else:
            self._obstacle_locations = []
            for _ in range(self.n_obstacles):
                while True:
                    obstacle_location_candidate = self._random_loc()
                    if np.array_equal(obstacle_location_candidate, self._agent_location):
                        continue
                    if np.array_equal(obstacle_location_candidate, self._target_location):
                        continue
                    self._obstacle_locations.append(obstacle_location_candidate)
                    break

    def _get_obs(self):
        obs = self._agent_location.tolist() + self._target_location.tolist()
        for obstacle_location in self._obstacle_locations:
            obs += obstacle_location.tolist()
        return obs

    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}
