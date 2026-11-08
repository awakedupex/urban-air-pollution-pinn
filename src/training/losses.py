"""Loss components of the PINN (all in dimensionless units).

L = w_d L_data + w_p L_phys + w_b L_bc + w_i L_ic + w_s L_src
"""
from __future__ import annotations

import torch

from src.physics.advection_diffusion import _grad


def data_loss(model, X, c):
    """Mean squared misfit at observation points (training stations only)."""
    return torch.mean((model(X) - c) ** 2)


def physics_loss(model, Xc):
    """Mean squared PDE residual at collocation points."""
    return torch.mean(model.residual(Xc) ** 2)


def boundary_loss(model, Xb, normals, c_bc=None):
    """Outflow faces (u·n > 0): zero normal gradient. Inflow faces: C' = c_bc if given.

    On real data the inflow concentration is unknown, so c_bc=None and only the
    weak outflow condition is imposed.
    """
    C = model(Xb)
    dC = _grad(C, Xb)[:, :2]
    u, v, _ = model.forcing.at(model.scales.t_rel(Xb[:, 2]))
    un = u * normals[:, 0] + v * normals[:, 1]
    out = un > 0
    terms = (torch.sum(dC * normals, dim=1) ** 2) * out
    if c_bc is not None:
        terms = terms + ((C - c_bc) ** 2) * (~out)
    return terms.mean()


def initial_loss(model, X0, c0):
    """Misfit to a known initial field at t' = -1 (synthetic twin only)."""
    return torch.mean((model(X0) - c0) ** 2)


def source_regulariser(model, Xc):
    """Smoothness of the emission field q and of the diurnal profile g.

    Without it the source can absorb any residual and the physics loss becomes vacuous.
    """
    Xs = Xc.detach().clone().requires_grad_(True)
    q = model.source.q(Xs)
    dq = _grad(q, Xs)[:, :2]
    g = model.source.g_nodes()
    curv = g.roll(-1) - 2 * g + g.roll(1)
    return torch.mean(dq ** 2) + torch.mean(curv ** 2)
