import torch


class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.center = None
        self.stdevs = None
        self.components = None

    def fit(self, x):
        self.center = torch.mean(x, dim=0)
        x_centered = x - self.center
        u, s, v = torch.svd(x_centered)
        self.components = v[:, :self.n_components]
        projected = x_centered @ self.components
        self.stdevs = torch.std(projected, dim=0)
        projected = projected / self.stdevs
        return projected

    def apply(self, x):
        x_centered = x - self.center
        projected = x_centered @ self.components
        projected = projected / self.stdevs
        return projected
