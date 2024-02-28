import random
import time
from copy import deepcopy
from dataclasses import dataclass
from pprint import pprint
from typing import Optional, Callable, List

import gymnasium as gym
import numpy as np

from src.dqn.dqn_agent import DQNAgent
from src.dqn.parameters import TrainParameters


@dataclass
class OptimizationResult:
    param: dict
    results: list
    score: float


class AgentOptimizer:

    def __init__(self, agent: DQNAgent) -> None:
        self.agent = agent

    def optimize(self,
                 base_train_param: TrainParameters,
                 train_param_options: dict,
                 env: gym.Env,
                 n_iter: int = 100,
                 score_func: Optional[Callable] = None) -> List[OptimizationResult]:

        if score_func is None:
            score_func = lambda results: np.mean(results[-50:])

        output = []
        for k in range(n_iter):
            self.agent.reset()

            train_param = deepcopy(base_train_param)
            selected_values = {}
            for field, values in train_param_options.items():
                value = random.choice(values)
                selected_values[field] = value
                setattr(train_param, field, value)

            pprint(selected_values)

            t1 = time.time()
            results = self.agent.train(env, train_param)
            t2 = time.time()
            duration = t2 - t1
            n_step_total = self.agent.steps_done
            print(f"Training took {duration} s for {n_step_total} steps, {n_step_total / duration:0.0f} steps/s")

            score = score_func(results)
            print(f"Iteration {k}, score {score}")

            output.append(OptimizationResult(param=selected_values, results=results, score=score))

        return output
