from abc import ABC, abstractmethod
from abc import ABC, abstractmethod
from typing import Any, Tuple, List, Union

import torch

from src.dqn.pca import PCA


class StateHashing(ABC):
    """
    Abstract interface for state hashing.
    """

    @abstractmethod
    def hash(self, states: torch.Tensor) -> List[Any]:
        """
        Calculate hashes for multiple states. States is rank 2 tensor with samples as rows. Output items should be
        hashable.
        """
        raise NotImplementedError


class ModelStateHashing(StateHashing):
    """
    Abstract interface for state hashing which requires model to fit.
    """

    @abstractmethod
    def fit(self, states: torch.Tensor) -> List[Tuple[int]]:
        """
        Fit internal model used for hashing. Returns hashes for states.
        """
        raise NotImplementedError


class RandomProjectionStateHashing(StateHashing):

    def __init__(self,
                 n_hashes: int,
                 n_dim: int = None,
                 scale_factors=1,
                 device="cuda") -> None:
        self.n_hashes = n_hashes
        self.scale_factor = scale_factors
        self.n_dim = n_dim

        self.projection_matrix = torch.randn(n_dim, n_hashes)
        self.projection_matrix /= torch.linalg.norm(self.projection_matrix, dim=0)
        self.projection_matrix = self.projection_matrix.to(device)

    def hash(self, x: torch.Tensor) -> List[Tuple[int]]:
        x_scaled = x / self.scale_factor
        projections = x_scaled @ self.projection_matrix
        values = torch.floor(projections).to(torch.int)
        return [tuple(row.tolist()) for row in values]


class PCAStateHashing(ModelStateHashing):

    def __init__(self, n_components=2, factor=1.0):
        self.pca = PCA(n_components)
        self.factor = factor

    def fit(self, states: torch.Tensor) -> List[Tuple[int]]:
        projections = self.pca.fit(states)
        hashes_tensor = torch.round(projections / self.factor).to(torch.int)
        return [tuple(row.tolist()) for row in hashes_tensor]

    def hash(self, states: torch.Tensor) -> List[Tuple[int]]:
        projections = self.pca.apply(states)
        hashes_tensor = torch.round(projections / self.factor).to(torch.int)
        return [tuple(row.tolist()) for row in hashes_tensor]
