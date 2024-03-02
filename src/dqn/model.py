import torch
import torch.nn as nn
import torch.nn.functional as F


class FNN(nn.Module):

    def __init__(self, n_observations: int, n_actions: int) -> None:
        super(FNN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.dropout12 = nn.Dropout(0.0)
        self.layer2 = nn.Linear(128, 128)
        self.dropout23 = nn.Dropout(0.0)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x: torch.Tensor, dropout: float = 0) -> torch.Tensor:
        """Calculate Q values for the states.
   
        Args:
            x (torch.Tensor): states, [n_samples x n_observations] tensor

        Returns:
            torch.Tensor: q values for actions, [n_samples x n_actions] tensor
        """
        self.dropout12.p = dropout
        self.dropout23.p = dropout

        x = F.relu(self.layer1(x))
        x = self.dropout12(x)
        x = F.relu(self.layer2(x))
        x = self.dropout23(x)
        return self.layer3(x)
