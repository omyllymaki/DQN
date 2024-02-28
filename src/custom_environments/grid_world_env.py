import gym
from gym import spaces
import numpy as np


class GridWorldEnv(gym.Env):

    def __init__(self, size=5, n_obstacles=3):
        self.size = size  # The size of the square grid
        self.n_obstacles = n_obstacles

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

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

    def _get_obs(self):
        obs = self._agent_location.tolist() + self._target_location.tolist()
        for obstacle_location in self._obstacle_locations:
            obs += obstacle_location.tolist()
        return obs

    def _get_info(self):
        return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        self._obstacle_locations = []
        for _ in range(self.n_obstacles):
            while True:
                obstacle_location_candidate = self.np_random.integers(0, self.size, size=2, dtype=int)
                if np.array_equal(obstacle_location_candidate, self._agent_location):
                    continue
                if np.array_equal(obstacle_location_candidate, self._target_location):
                    continue
                self._obstacle_locations.append(obstacle_location_candidate)
                break

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        terminated = False
        reward = 0
        if np.array_equal(self._agent_location, self._target_location):  # Agent reached the target
            terminated = True
            reward = 1
        for obstacle_location in self._obstacle_locations:
            if np.array_equal(self._agent_location, obstacle_location):  # Agent hits to obstacle
                terminated = True
                reward = -1

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info
