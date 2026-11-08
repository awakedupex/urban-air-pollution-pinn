"""Tests for the physics core (src/physics).

The PDE is
    C_t + u C_x + v C_y = D (C_xx + C_yy) - lam C + S.

Independent checks:
  * autograd derivatives vs closed-form derivatives of a manufactured function;
  * residual of a manufactured solution equals the hand-computed residual;
  * the Gaussian puff (exact solution, S = 0) has ~zero residual;
  * non-dimensionalisation preserves the residual up to the scale factor T/C_ref;
  * the finite-volume reference solver reproduces exact solutions
    (pure decay, approach to equilibrium S/lam, advected-diffusing puff),
    conserves mass, and converges under grid refinement.
"""
import numpy as np
import pytest
import torch

from src.physics.advection_diffusion import (
    adr_residual,
    gaussian_puff,
    gradients,
    nondimensional_coefficients,
)
from src.physics.solver import Grid, solve_adr, stable_dt

torch.set_default_dtype(torch.float64)


def _points(n=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 3, generator=g).requires_grad_(True)


# ------------------------------------------------------------------ autograd
def test_autograd_matches_analytic_derivatives():
    X = _points()
    x, y, t = X[:, 0], X[:, 1], X[:, 2]
    C = torch.sin(2 * x) * torch.cos(3 * y) * torch.exp(-t)
    d = gradients(C, X)
    sx, cx = torch.sin(2 * x), torch.cos(2 * x)
    sy, cy = torch.sin(3 * y), torch.cos(3 * y)
    e = torch.exp(-t)
    torch.testing.assert_close(d["t"], -sx * cy * e)
    torch.testing.assert_close(d["x"], 2 * cx * cy * e)
    torch.testing.assert_close(d["y"], -3 * sx * sy * e)
    torch.testing.assert_close(d["xx"], -4 * sx * cy * e)
    torch.testing.assert_close(d["yy"], -9 * sx * cy * e)


def test_residual_manufactured_solution():
    X = _points()
    x, y, t = X[:, 0], X[:, 1], X[:, 2]
    C = torch.sin(2 * x) * torch.cos(3 * y) * torch.exp(-t)
    u, v, D, lam = 0.7, -0.4, 0.05, 0.3
    S = torch.full_like(x, 0.2)
    r = adr_residual(C, X, u, v, D, lam, S)
    sx, cx, sy, cy, e = torch.sin(2 * x), torch.cos(2 * x), torch.sin(3 * y), torch.cos(3 * y), torch.exp(-t)
    expected = (-sx * cy * e) + u * (2 * cx * cy * e) + v * (-3 * sx * sy * e) \
        - D * (-13 * sx * cy * e) + lam * (sx * cy * e) - 0.2
    torch.testing.assert_close(r, expected.detach())


def test_residual_accepts_per_point_coefficients():
    X = _points()
    C = X[:, 0] * 2.0  # C_x = 2, everything else 0
    u = torch.linspace(-1, 1, len(X))
    r = adr_residual(C, X, u, torch.zeros_like(u), 0.0, 0.0, torch.zeros_like(u))
    torch.testing.assert_close(r, 2.0 * u)


def test_gaussian_puff_is_exact_solution():
    X = (torch.rand(500, 3, dtype=torch.float64) * torch.tensor([4.0, 4.0, 1.0]) + torch.tensor([-2.0, -2.0, 0.5]))
    X.requires_grad_(True)
    u, v, D, lam = 0.8, 0.3, 0.1, 0.2
    C = gaussian_puff(X[:, 0], X[:, 1], X[:, 2], M=1.0, x0=0.0, y0=0.0, u=u, v=v, D=D, lam=lam)
    r = adr_residual(C, X, u, v, D, lam, torch.zeros(len(X)))
    assert r.abs().max().item() < 1e-10


def test_nondimensionalisation_scales_residual():
    """Residual in scaled variables == (T / C_ref) * residual in physical variables."""
    L, T, Cref = 2.0e4, 3.6e3, 100.0
    u, v, D, lam = 2.0, -1.0, 300.0, 5e-6
    puff = dict(M=1e9, x0=0.0, y0=0.0, u=u, v=v, D=D, lam=lam)

    Xp = torch.tensor([[1000.0, -2000.0, 7200.0], [-500.0, 300.0, 9000.0]], requires_grad=True)
    Sphys = torch.tensor([0.01, 0.02])
    rp = adr_residual(gaussian_puff(Xp[:, 0], Xp[:, 1], Xp[:, 2], **puff), Xp, u, v, D, lam, Sphys)

    Xs = (Xp.detach() / torch.tensor([L, L, T])).requires_grad_(True)
    Cs = gaussian_puff(Xs[:, 0] * L, Xs[:, 1] * L, Xs[:, 2] * T, **puff) / Cref
    k = nondimensional_coefficients(u=u, v=v, D=D, lam=lam, S=Sphys, L=L, T=T, C_ref=Cref)
    rs = adr_residual(Cs, Xs, k["u"], k["v"], k["D"], k["lam"], k["S"])
    torch.testing.assert_close(rs, rp * T / Cref)


# ------------------------------------------------------------------ solver
def _grid(n=81, half=10_000.0):
    return Grid(x_min=-half, x_max=half, y_min=-half, y_max=half, nx=n, ny=n)


def test_stable_dt_respects_cfl_and_diffusion_limits():
    g = _grid()
    dt = stable_dt(g, umax=3.0, vmax=1.0, D=500.0, cfl=0.5)
    assert dt * (3.0 / g.dx + 1.0 / g.dy) <= 0.5 + 1e-12
    assert dt * 500.0 * (1 / g.dx**2 + 1 / g.dy**2) <= 0.25 + 1e-12


def test_solver_pure_decay():
    g = _grid(n=21)
    lam, C0, T = 1e-4, 50.0, 7200.0
    out = solve_adr(
        g, C0=np.full(g.shape, C0), t_end=T, u=lambda t: 0.0, v=lambda t: 0.0, D=0.0, lam=lam,
        S=lambda X, Y, t: 0.0, C_bc=lambda t: C0 * np.exp(-lam * t), save_every=None,
    )
    np.testing.assert_allclose(out["C"][-1], C0 * np.exp(-lam * T), rtol=1e-6)


def test_solver_approaches_equilibrium():
    """Uniform source and removal: C(t) = S/lam (1 - e^{-lam t}) -> equilibrium C* = S/lam."""
    g = _grid(n=21)
    lam, S0, T = 1e-4, 2e-3, 50_000.0
    exact = lambda t: S0 / lam * (1 - np.exp(-lam * t))
    out = solve_adr(
        g, C0=np.zeros(g.shape), t_end=T, u=lambda t: 1.5, v=lambda t: -0.5, D=100.0, lam=lam,
        S=lambda X, Y, t: S0, C_bc=exact, save_every=None,
    )
    np.testing.assert_allclose(out["C"][-1], exact(T), rtol=1e-5)
    assert abs(out["C"][-1].mean() - S0 / lam) / (S0 / lam) < 0.01


def _puff_case(n):
    # Released at (-12, -8) km; centre at t0 = 1 h is (-4.8, -4.4) km and at
    # t0 + T = 2.5 h is (6, 1) km, > 3 sigma from every boundary of the 30 km box.
    g = _grid(n=n, half=15_000.0)
    u, v, D, lam = 2.0, 1.0, 400.0, 1e-5
    t0, T = 3600.0, 5400.0
    puff = lambda X, Y, t: gaussian_puff(X, Y, t, M=1e10, x0=-12000.0, y0=-8000.0, u=u, v=v, D=D, lam=lam)
    out = solve_adr(
        g, C0=puff(g.X, g.Y, t0), t_end=T, u=lambda t: u, v=lambda t: v, D=D, lam=lam,
        S=lambda X, Y, t: 0.0, C_bc=lambda t: 0.0, save_every=None, t0=t0,
    )
    exact = puff(g.X, g.Y, t0 + T)
    return g, out["C"][-1], exact


def test_solver_matches_gaussian_puff():
    g, C, exact = _puff_case(161)
    rel = np.linalg.norm(C - exact) / np.linalg.norm(exact)
    assert rel < 0.02


def test_solver_converges_under_refinement():
    errs = []
    for n in (41, 81, 161):
        g, C, exact = _puff_case(n)
        errs.append(np.linalg.norm(C - exact) / np.linalg.norm(exact))
    assert errs[0] / errs[1] > 2.5 and errs[1] / errs[2] > 2.5


def test_solver_conserves_mass_without_removal():
    g = _grid(n=81)
    D = 300.0
    puff = lambda X, Y, t: gaussian_puff(X, Y, t, M=1e10, x0=0.0, y0=0.0, u=0.0, v=0.0, D=D, lam=0.0)
    C0 = puff(g.X, g.Y, 3600.0)
    out = solve_adr(
        g, C0=C0, t_end=3600.0, u=lambda t: 0.5, v=lambda t: 0.0, D=D, lam=0.0,
        S=lambda X, Y, t: 0.0, C_bc=lambda t: 0.0, save_every=None, t0=3600.0,
    )
    m0, m1 = C0.sum(), out["C"][-1].sum()
    assert abs(m1 - m0) / m0 < 1e-3
