"""Helpers for the synthetic twin (all values produced here are SYNTHETIC)."""
from __future__ import annotations

import numpy as np


def bilinear_sample(grid, field, x, y):
    """Bilinear interpolation of a grid field at points (x, y) inside the grid."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if (x < grid.x_min).any() or (x > grid.x_max).any() or (y < grid.y_min).any() or (y > grid.y_max).any():
        raise ValueError("sample point outside grid")
    fx = (x - grid.x_min) / grid.dx
    fy = (y - grid.y_min) / grid.dy
    i = np.clip(np.floor(fx).astype(int), 0, grid.nx - 2)
    j = np.clip(np.floor(fy).astype(int), 0, grid.ny - 2)
    a, b = fx - i, fy - j
    return ((1 - a) * (1 - b) * field[i, j] + a * (1 - b) * field[i + 1, j]
            + (1 - a) * b * field[i, j + 1] + a * b * field[i + 1, j + 1])


def diurnal_profile(local_hour):
    """Traffic-like emission profile: morning (09 h) and evening (20 h) peaks, mean 1 over a day."""
    h = np.asarray(local_hour, float)
    bump = lambda c, w: np.exp(-0.5 * (((h - c + 12) % 24 - 12) / w) ** 2)
    raw = 0.5 + 0.8 * bump(9.0, 2.0) + 1.0 * bump(20.0, 2.5)
    ref = 0.5 + 0.8 * bump_mean(9.0, 2.0) + 1.0 * bump_mean(20.0, 2.5)
    return raw / ref


def bump_mean(c, w):
    hh = np.arange(24.0)
    return np.mean(np.exp(-0.5 * (((hh - c + 12) % 24 - 12) / w) ** 2))
