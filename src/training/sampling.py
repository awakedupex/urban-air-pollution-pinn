"""Sampling of collocation, boundary and initial points in the dimensionless cube [-1, 1]^3."""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import qmc


def sample_collocation(n, seed=0) -> torch.Tensor:
    """Latin hypercube sample over (x', y', t') in [-1, 1]^3."""
    X = 2 * qmc.LatinHypercube(d=3, seed=seed).random(n) - 1
    return torch.as_tensor(X, dtype=torch.get_default_dtype())


def sample_boundary(n, seed=0):
    """Points on the four lateral faces x' = ±1, y' = ±1 with outward unit normals."""
    rng = np.random.default_rng(seed)
    face = rng.integers(0, 4, n)
    s, t = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
    X = np.empty((n, 3))
    N = np.zeros((n, 2))
    sign = np.where(face % 2 == 0, -1.0, 1.0)
    on_x = face < 2
    X[:, 0] = np.where(on_x, sign, s)
    X[:, 1] = np.where(on_x, s, sign)
    X[:, 2] = t
    N[on_x, 0] = sign[on_x]
    N[~on_x, 1] = sign[~on_x]
    f = lambda a: torch.as_tensor(a, dtype=torch.get_default_dtype())
    return f(X), f(N)


def sample_initial(n, seed=0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.uniform(-1, 1, n), rng.uniform(-1, 1, n), -np.ones(n)])
    return torch.as_tensor(X, dtype=torch.get_default_dtype())
