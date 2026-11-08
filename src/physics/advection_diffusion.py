"""Advection-diffusion-reaction (ADR) operator for the PINN.

Governing equation (mixing-layer-averaged PM2.5, 2D + time):

    C_t + u(t) C_x + v(t) C_y = D (C_xx + C_yy) - lam C + S(x, y, t)

The residual  r = C_t + u C_x + v C_y - D (C_xx + C_yy) + lam C - S  is computed
with automatic differentiation. Coefficients may be scalars or per-point tensors,
and must be expressed in the same units as the coordinates used for X.
"""
from __future__ import annotations

import numpy as np
import torch


def _grad(f: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    # A derivative that is constant in X (e.g. of a linear field) is not connected
    # to X in the graph; its gradient is then identically zero.
    if not f.requires_grad:
        return torch.zeros_like(X)
    g = torch.autograd.grad(f, X, grad_outputs=torch.ones_like(f), create_graph=True, allow_unused=True)[0]
    return torch.zeros_like(X) if g is None else g


def gradients(C: torch.Tensor, X: torch.Tensor) -> dict:
    """First derivatives and the two pure second space derivatives of C(X), X = (x, y, t)."""
    dC = _grad(C, X)
    C_x, C_y, C_t = dC[:, 0], dC[:, 1], dC[:, 2]
    C_xx = _grad(C_x, X)[:, 0]
    C_yy = _grad(C_y, X)[:, 1]
    return {"x": C_x, "y": C_y, "t": C_t, "xx": C_xx, "yy": C_yy}


def adr_residual(C, X, u, v, D, lam, S) -> torch.Tensor:
    """Pointwise PDE residual r(x, y, t); zero for an exact solution."""
    d = gradients(C, X)
    return d["t"] + u * d["x"] + v * d["y"] - D * (d["xx"] + d["yy"]) + lam * C - S


def nondimensional_coefficients(u, v, D, lam, S, L, T, C_ref) -> dict:
    """Coefficients of the PDE in variables x' = x/L, y' = y/L, t' = t/T, C' = C/C_ref.

    Dividing the physical PDE by C_ref/T gives
        C'_t' + (uT/L) C'_x' + (vT/L) C'_y' = (DT/L^2) lap' C' - (lam T) C' + S T / C_ref,
    i.e. the groups uT/L (advective Courant-like number), DT/L^2 (= (uT/L)/Pe)
    and lam T (Damkohler-like number).
    """
    return {"u": u * T / L, "v": v * T / L, "D": D * T / L**2, "lam": lam * T, "S": S * T / C_ref}


def gaussian_puff(x, y, t, M, x0, y0, u, v, D, lam):
    """Exact solution for an instantaneous release of mass M (per unit depth) at t = 0.

        C = M / (4 pi D t) * exp(-[(x - x0 - u t)^2 + (y - y0 - v t)^2] / (4 D t) - lam t)

    Works with NumPy arrays or torch tensors.
    """
    lib = torch if isinstance(t, torch.Tensor) else np
    r2 = (x - x0 - u * t) ** 2 + (y - y0 - v * t) ** 2
    return M / (4 * np.pi * D * t) * lib.exp(-r2 / (4 * D * t) - lam * t)
