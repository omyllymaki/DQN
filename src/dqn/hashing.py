import math
from abc import ABC, abstractmethod
from typing import Any, Tuple, List, Optional, Union

import numpy as np
import torch


class Hashing(ABC):
    """
    Abstract interface for hashing.
    """

    @abstractmethod
    def apply(self, *args) -> Any:
        """
        Calculate hash of the inputs
        """
        raise NotImplementedError


class RewardHashing(Hashing):
    """
    Hash reward.
    """

    def __init__(self, resolution: float) -> None:
        self.resolution = resolution

    def apply(self, reward: torch.Tensor) -> int:
        return np.round(reward.item() / self.resolution).astype(int)


class RandomProjectionHashing:

    def __init__(self,
                 n_hashes: int,
                 n_dim: int,
                 scale_factors=1,
                 device="cuda") -> None:
        self.n_hashes = n_hashes
        self.scale_factor = scale_factors
        self.n_dim = n_dim

        self.projection_matrix = torch.randn(n_dim, n_hashes)
        self.projection_matrix /= torch.linalg.norm(self.projection_matrix, dim=0)
        self.projection_matrix = self.projection_matrix.to(device)

    def apply(self, x: torch.Tensor) -> Union[Tuple[int], List[Tuple[int]]]:
        x_scaled = x / self.scale_factor
        projections = x_scaled @ self.projection_matrix
        values = torch.floor(projections).to(torch.int)
        if (values.shape[0] == 1) or (len(values.shape) == 1):
            return tuple(values.flatten().tolist())
        else:
            return [tuple(row.tolist()) for row in values]
