"""Physics-informed neural network for mixing-layer-averaged PM2.5.

Dimensionless variables (inputs of every network lie in [-1, 1]):
    x' = (x - x_c)/L,   y' = (y - y_c)/L,   t' = 2 t_rel / T - 1,   C' = C / C_ref,
where t_rel is seconds since the window start. With dt = (T/2) dt', the PDE

    C_t + u C_x + v C_y = D lap C - lam C + E(x, y) g(tau) / H(t)

becomes  C'_t' + U C'_x' + V C'_y' = K lap' C' - Lam C' + S'  with
    U = u T/(2L),  V = v T/(2L),  K = D T/(2L^2),  Lam = lam T/2,
    S' = q(x', y') g(tau) / h(t),   h = max(H, H_min)/H_ref.
The emission magnitude is absorbed into the learned field q >= 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.physics.advection_diffusion import adr_residual, nondimensional_coefficients

IST_OFFSET_H = 5.5


@dataclass
class Scales:
    x_c: float  # m
    y_c: float  # m
    L: float  # m, half-width of the domain
    t_start: pd.Timestamp  # UTC, t' = -1
    T: float  # s, window length, t' = +1 at t_start + T
    C_ref: float  # µg/m³

    def inputs(self, x_m, y_m, t_rel_s) -> torch.Tensor:
        X = np.stack([(np.asarray(x_m) - self.x_c) / self.L,
                      (np.asarray(y_m) - self.y_c) / self.L,
                      2 * np.asarray(t_rel_s) / self.T - 1], axis=1)
        return torch.as_tensor(X, dtype=torch.get_default_dtype())

    def t_rel(self, t_nd: torch.Tensor) -> torch.Tensor:
        return (t_nd + 1) * self.T / 2

    def local_hour(self, t_nd: torch.Tensor) -> torch.Tensor:
        h0 = self.t_start.hour + self.t_start.minute / 60 + IST_OFFSET_H
        return torch.remainder(h0 + self.t_rel(t_nd) / 3600.0, 24.0)


class Forcing(nn.Module):
    """Hourly ERA5 wind (m/s) and boundary-layer height, linearly interpolated in time."""

    def __init__(self, t_rel_s, u, v, H, H_min=50.0, H_ref=500.0):
        super().__init__()
        f = lambda a: torch.tensor(np.array(a, dtype=float), dtype=torch.get_default_dtype())
        self.register_buffer("t", f(t_rel_s))
        self.register_buffer("u", f(u))
        self.register_buffer("v", f(v))
        self.register_buffer("h", f(np.maximum(H, H_min) / H_ref))

    def at(self, t_rel_s: torch.Tensor):
        t = t_rel_s.detach().to(self.t.dtype).clamp(self.t[0], self.t[-1])
        i = torch.searchsorted(self.t, t, right=True).clamp(1, len(self.t) - 1)
        w = (t - self.t[i - 1]) / (self.t[i] - self.t[i - 1])
        lerp = lambda a: a[i - 1] + w * (a[i] - a[i - 1])
        return lerp(self.u), lerp(self.v), lerp(self.h)


def mlp(n_in, n_out, width, depth) -> nn.Sequential:
    layers, d = [], n_in
    for _ in range(depth):
        lin = nn.Linear(d, width)
        nn.init.xavier_normal_(lin.weight)
        nn.init.zeros_(lin.bias)
        layers += [lin, nn.Tanh()]
        d = width
    out = nn.Linear(d, n_out)
    nn.init.xavier_normal_(out.weight)
    nn.init.zeros_(out.bias)
    return nn.Sequential(*layers, out)


class ConcentrationNet(nn.Module):
    """(x', y', t') -> C' >= 0 (softplus output). tanh keeps second derivatives non-zero."""

    def __init__(self, width=64, depth=5):
        super().__init__()
        self.body = mlp(3, 1, width, depth)

    def forward(self, X):
        return nn.functional.softplus(self.body(X))[:, 0]


class SourceModel(nn.Module):
    """S'(x', y', t') = q(x', y') g(tau) / h(t): spatial emission x diurnal profile / mixing height."""

    def __init__(self, scales: Scales, forcing: Forcing, width=32, depth=2):
        super().__init__()
        self.scales, self.forcing = scales, forcing
        self.q_net = mlp(2, 1, width, depth)
        self.g_raw = nn.Parameter(torch.zeros(24))

    def q(self, X):
        return nn.functional.softplus(self.q_net(X[:, :2]))[:, 0]

    def g_nodes(self):
        g = nn.functional.softplus(self.g_raw)
        return g / g.mean()

    def g(self, hours):
        """Periodic piecewise-linear interpolation of the 24 hourly nodes."""
        nodes = self.g_nodes()
        h = torch.remainder(hours, 24.0)
        i0 = torch.floor(h).long() % 24
        w = h - torch.floor(h)
        return (1 - w) * nodes[i0] + w * nodes[(i0 + 1) % 24]

    def forward(self, X):
        _, _, h = self.forcing.at(self.scales.t_rel(X[:, 2]))
        return self.q(X) * self.g(self.scales.local_hour(X[:, 2])) / h


class PINN(nn.Module):
    def __init__(self, scales: Scales, forcing: Forcing, D_min, D_max, D_init, lam,
                 width=64, depth=5, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.scales, self.forcing, self.lam = scales, forcing, float(lam)
        self.D_min, self.D_max = float(D_min), float(D_max)
        self.net = ConcentrationNet(width, depth)
        self.source = SourceModel(scales, forcing)
        p = (D_init - D_min) / (D_max - D_min)
        self.D_raw = nn.Parameter(torch.tensor(float(np.log(p / (1 - p)))))

    def D(self):
        """Eddy diffusivity (m²/s), constrained to [D_min, D_max] by a sigmoid."""
        return self.D_min + (self.D_max - self.D_min) * torch.sigmoid(self.D_raw)

    def forward(self, X):
        return self.net(X)

    def residual(self, X):
        """Dimensionless PDE residual at collocation points X (requires_grad=True)."""
        C = self.net(X)
        u, v, _ = self.forcing.at(self.scales.t_rel(X[:, 2]))
        k = nondimensional_coefficients(u=u, v=v, D=self.D(), lam=self.lam, S=0.0,
                                        L=self.scales.L, T=self.scales.T / 2, C_ref=1.0)
        return adr_residual(C, X, k["u"], k["v"], k["D"], k["lam"], self.source(X))
