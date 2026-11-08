"""Reference finite-difference solver for the 2D ADR equation.

Used (i) to generate the synthetic-twin data set, (ii) as the "numerical
simulation" component of the report, and (iii) as an independent check of the
physics implemented in the PINN.

Discretisation (method of lines on a uniform cell-centred grid):
  * advection  — third-order upwind-biased differences (low numerical diffusion);
  * diffusion  — second-order central differences;
  * time       — strong-stability-preserving Runge-Kutta 3 (SSP-RK3);
  * boundaries — inflow faces: Dirichlet C = C_bc(t); outflow faces: zero gradient.
Stability: dt is limited by the advective CFL number and the diffusive number
D dt (1/dx^2 + 1/dy^2) <= 1/4 (see `stable_dt`; von Neumann analysis in report).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

G = 2  # ghost cells per side


@dataclass
class Grid:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    nx: int
    ny: int
    x: np.ndarray = field(init=False)
    y: np.ndarray = field(init=False)

    def __post_init__(self):
        self.x = np.linspace(self.x_min, self.x_max, self.nx)
        self.y = np.linspace(self.y_min, self.y_max, self.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

    @property
    def dx(self):
        return self.x[1] - self.x[0]

    @property
    def dy(self):
        return self.y[1] - self.y[0]

    @property
    def shape(self):
        return (self.nx, self.ny)


def stable_dt(grid: Grid, umax, vmax, D, cfl=0.5, diff_number=0.25) -> float:
    adv = abs(umax) / grid.dx + abs(vmax) / grid.dy
    dif = D * (1 / grid.dx**2 + 1 / grid.dy**2)
    limits = [cfl / adv if adv > 0 else np.inf, diff_number / dif if dif > 0 else np.inf]
    return float(min(limits))


def _pad(C, u, v, cb):
    """Fill ghost cells: Dirichlet on inflow faces, zero-gradient on outflow faces."""
    P = np.pad(C, G, mode="edge")
    if u > 0:
        P[:G, :] = cb
    elif u < 0:
        P[-G:, :] = cb
    if v > 0:
        P[:, :G] = cb
    elif v < 0:
        P[:, -G:] = cb
    return P


def _upwind3(P, vel, h, axis):
    """Third-order upwind-biased first derivative on the interior of padded P."""
    s = lambda k: np.roll(P, -k, axis=axis)  # s(k)[i] = P[i + k]
    if vel >= 0:
        d = (2 * s(1) + 3 * P - 6 * s(-1) + s(-2)) / (6 * h)
    else:
        d = (-s(2) + 6 * s(1) - 3 * P - 2 * s(-1)) / (6 * h)
    return d[G:-G, G:-G]


def _rhs(C, t, grid, u, v, D, lam, S, C_bc):
    ut, vt, cb = u(t), v(t), C_bc(t)
    P = _pad(C, ut, vt, cb)
    adv = ut * _upwind3(P, ut, grid.dx, 0) + vt * _upwind3(P, vt, grid.dy, 1)
    lap = (P[G + 1:-G + 1 or None, G:-G] - 2 * C + P[G - 1:-G - 1, G:-G]) / grid.dx**2 \
        + (P[G:-G, G + 1:-G + 1 or None] - 2 * C + P[G:-G, G - 1:-G - 1]) / grid.dy**2
    return -adv + D * lap - lam * C + S(grid.X, grid.Y, t)


def solve_adr(grid, C0, t_end, u, v, D, lam, S, C_bc, save_every=3600.0, t0=0.0,
              umax=None, vmax=None, cfl=0.5):
    """Integrate from t0 to t0 + t_end. u, v, C_bc are functions of t; S of (X, Y, t).

    Returns dict with 't' (saved times) and 'C' (array [n_saved, nx, ny]); the
    final state is always saved.
    """
    if umax is None or vmax is None:
        ts = np.linspace(t0, t0 + t_end, 200)
        umax = max(abs(u(s)) for s in ts) if umax is None else umax
        vmax = max(abs(v(s)) for s in ts) if vmax is None else vmax
    dt_max = stable_dt(grid, umax, vmax, D, cfl=cfl)
    if not np.isfinite(dt_max):
        dt_max = t_end / 100

    f = lambda C, t: _rhs(C, t, grid, u, v, D, lam, S, C_bc)
    C, t = np.array(C0, dtype=float), t0
    times, states = [], []
    next_save = t0 + save_every if save_every else np.inf
    if save_every:
        times.append(t), states.append(C.copy())
    t_stop = t0 + t_end
    while t < t_stop - 1e-9:
        dt = min(dt_max, t_stop - t, next_save - t)
        k1 = C + dt * f(C, t)
        k2 = 0.75 * C + 0.25 * (k1 + dt * f(k1, t + dt))
        C = C / 3 + 2 / 3 * (k2 + dt * f(k2, t + 0.5 * dt))
        t += dt
        if save_every and abs(t - next_save) < 1e-9:
            times.append(t), states.append(C.copy())
            next_save += save_every
    if not times or times[-1] != t:
        times.append(t), states.append(C.copy())
    return {"t": np.array(times), "C": np.array(states)}
